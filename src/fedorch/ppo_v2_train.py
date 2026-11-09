"""GPU-only controller learning over immutable frozen-tool replay caches.

Supervised mode is an exploratory observed-action Monte-Carlo cost regressor,
not an exhaustive action oracle. Audit IPW is supplied by the environment only
on verified error; all observed operational costs remain unweighted.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import random
import time
import resource
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .ppo_v2_env import ACTIONS, FEATURES, Bundle, InspectionEnv

SCHEMA = 'fedorch-ppo-v2-actor-v1'
CAP_BYTES = 30_000_000


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed signed-log compression handles mixed physical units without
        # fitting statistics to any client or held-out split.
        self.net = nn.Sequential(nn.Linear(len(FEATURES), 192), nn.Tanh(),
                                 nn.Linear(192, 192), nn.Tanh(), nn.Linear(192, len(ACTIONS)))
        if sum(p.numel() for p in self.parameters()) > 100_000:
            raise ValueError('Actor exceeds 100,000 parameters')

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0., posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        return self.net(torch.sign(x) * torch.log1p(torch.abs(x)))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURES), 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0., posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        return self.net(torch.sign(x) * torch.log1p(torch.abs(x))).squeeze(-1)


def masked_distribution(logits, mask):
    if not bool(mask.any(dim=-1).all()):
        raise ValueError('Every stored mask must contain a legal action')
    return Categorical(logits=logits.masked_fill(~mask.bool(), -torch.inf))


def duration_gae(rewards, values, durations, dones, kappa=.01, lam=.95):
    """Terminal-aware semi-Markov GAE; values contains final bootstrap value."""
    if len(values) != len(rewards) + 1:
        raise ValueError('One bootstrap value required')
    advantages = [0.] * len(rewards)
    running = 0.
    for i in reversed(range(len(rewards))):
        gamma = math.exp(-kappa * durations[i])
        alive = 1. - float(dones[i])
        delta = rewards[i] + gamma * values[i + 1] * alive - values[i]
        running = delta + gamma * lam * alive * running
        advantages[i] = running
    return advantages, [a + v for a, v in zip(advantages, values)]


def ppo_clip_loss(logits, masks, actions, old_log_probs, advantages, clip=.2):
    distribution = masked_distribution(logits, masks)
    ratio = (distribution.log_prob(actions) - old_log_probs).exp()
    objective = torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages)
    return -objective.mean(), distribution.entropy().mean()


def audit_cost(known_cost, verified_error, audited, propensity):
    """Explicit guard for consumers constructing supervised targets."""
    if not audited:
        if verified_error is not None:
            raise ValueError('Unverified error must remain missing')
        return known_cost
    if not 0 < propensity <= 1:
        raise ValueError('Invalid audit propensity')
    return known_cost + (0. if verified_error is None else verified_error / propensity)


def serialize_delta(actor, reference):
    payload = io.BytesIO()
    delta = {key: (value.detach().cpu() - reference[key].cpu()).half()
             for key, value in actor.state_dict().items()}
    torch.save(delta, payload)
    return payload.getvalue(), delta


def preflight_payloads(payloads, cumulative=0, cap=CAP_BYTES):
    total = cumulative + sum(len(p) for p in payloads)
    if total >= cap:
        raise ValueError(f'Serialized actor payload would reach {total} bytes; strict cap is <{cap}')
    return total


def fedavg(reference, deltas, weights):
    if not deltas or sum(weights) <= 0:
        raise ValueError('No participating client weight')
    return {key: reference[key] + sum(delta[key].float() * (weight / sum(weights))
                                    for delta, weight in zip(deltas, weights))
            for key in reference}


@dataclass
class Rollout:
    states: list
    masks: list
    actions: list
    log_probs: list
    rewards: list
    durations: list
    dones: list
    values: list
    version: int
    records: list = field(default_factory=list)


def heuristic_action(state, mask):
    # Label-free fixed cascade: optional explanation once, then accept; masking
    # prevents repeats. Never reads labels or assessment of correctness.
    for name in ('SMOL_EXPLAIN', 'ACCEPT', 'ABSTAIN'):
        if name in ACTIONS and mask[ACTIONS.index(name)]:
            return ACTIONS.index(name)
    return int(np.flatnonzero(mask)[0])


def mission_items(pool, budget, rng):
    eligible = pool
    if budget >= 15:
        tower_sources = {}
        for item in pool: tower_sources.setdefault(item['tower'], set()).add(item['source_id'])
        eligible = [x for x in pool if len(tower_sources[x['tower']]) >= 2]
        if not eligible: raise ValueError('15-second missions require same-tower distinct-source groups')
    first = eligible[rng.randrange(len(eligible))]
    if budget < 15:
        return [first]
    candidates = [x for x in pool if x['tower'] == first['tower'] and x['source_id'] != first['source_id']]
    rng.shuffle(candidates)
    result, seen = [first], {first['source_id']}
    target = rng.randint(2, 5)
    for item in candidates:
        if item['source_id'] not in seen:
            result.append(item); seen.add(item['source_id'])
        if len(result) == target:
            break
    return result


def evaluation_missions(items, budget):
    """Deterministic 2--3 item same-tower groups; disclose ungroupable IDs."""
    if budget < 15:
        return [[item] for item in items], []
    towers = {}
    for item in items:
        towers.setdefault(item['tower'], {}).setdefault(item['source_id'], []).append(item)
    groups, excluded = [], []
    for sources in towers.values():
        tower_groups = []
        while sum(bool(queue) for queue in sources.values()) >= 2:
            ordered = sorted((source for source in sources if sources[source]),
                             key=lambda source: (-len(sources[source]), source))
            tower_groups.append([sources[source].pop(0) for source in ordered[:2]])
        for queue in sources.values():
            for item in queue:
                target = next((group for group in tower_groups if len(group)<5 and
                               item['source_id'] not in {x['source_id'] for x in group}), None)
                if target is None: excluded.append(item['item_id'])
                else: target.append(item)
        groups.extend(tower_groups)
    return groups, excluded


def collect(bundle, pool, actor, critic, args, version, seed, episodes, behavior=None):
    mission_rng = random.Random(seed)
    action_rng = random.Random(seed + 1_000_000_007)
    batch = Rollout([], [], [], [], [], [], [], [], version)
    results = []
    for episode in range(episodes):
        budget = args.budgets[episode % len(args.budgets)]
        env = InspectionEnv(bundle, mission_items(pool, budget, mission_rng), budget_s=budget,
                            feedback=args.feedback, seed=seed+episode, policy_version=version,
                            allow_enhance=args.enhance, network=getattr(args, 'network', None))
        state = env.reset()
        done = False
        while not done:
            if args.device == 'cuda': torch.cuda.synchronize()
            started = time.perf_counter()
            mask = np.asarray(env.mask(), dtype=bool).copy()
            tensor = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=args.device)
            with torch.no_grad():
                logits = actor(tensor)
                dist = masked_distribution(logits, torch.as_tensor(mask, device=args.device))
                if behavior == 'heuristic':
                    action = heuristic_action(state, mask)
                elif behavior == 'supervised':
                    action = (action_rng.choice(np.flatnonzero(mask).tolist()) if action_rng.random() < .2
                              else int(logits.masked_fill(~torch.as_tensor(mask, device=args.device), -torch.inf).argmax()))
                else:
                    action = int(dist.sample())
                logp = float(dist.log_prob(torch.tensor(action, device=args.device)))
            if args.device == 'cuda': torch.cuda.synchronize()
            duration = time.perf_counter() - started
            # Training-only critic execution is outside deployed policy timing.
            with torch.no_grad(): value = float(critic(tensor))
            next_state, reward, done, info = env.step(action, policy_duration_s=duration)
            if info.get('policy_version', version) != version:
                raise ValueError('Environment returned stale policy version')
            batch.records.append({'item_id': info.get('item_id'), 'mission_id': f'{seed}:{episode}',
                                  'policy_version': version, 'audited': info.get('audited'),
                                  'audit_propensity': info.get('audit_propensity'),
                                  'observed_error': info.get('observed_error')})
            batch.states.append(np.asarray(state, dtype=np.float32).copy())
            batch.masks.append(mask); batch.actions.append(action); batch.log_probs.append(logp)
            batch.rewards.append(float(reward)); batch.durations.append(float(info['duration_s']))
            batch.dones.append(done); batch.values.append(value)
            state = next_state
        results.extend(env.results)
    batch.values.append(0.)
    return batch, results


def merge_rollouts(batches, version):
    """Join completed on-policy batches without inserting bootstrap states."""
    combined = Rollout([], [], [], [], [], [], [], [], version)
    for batch in batches:
        if batch.version != version:
            raise ValueError('Cannot merge stale policy versions')
        if batch.dones and not batch.dones[-1]:
            raise ValueError('Only completed mission batches can be merged')
        for name in ('states', 'masks', 'actions', 'log_probs', 'rewards',
                     'durations', 'dones', 'records'):
            getattr(combined, name).extend(getattr(batch, name))
        combined.values.extend(batch.values[:-1])
    combined.values.append(0.)
    return combined


def update(actor, critic, batch, version, args, supervised=False):
    if batch.version != version:
        raise ValueError('Stale rollout rejected after actor federation/update')
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Training is CUDA-only; no CPU training is permitted')
    if not batch.states:
        return {}
    device = args.device
    states = torch.as_tensor(np.asarray(batch.states), device=device)
    masks = torch.as_tensor(np.asarray(batch.masks), device=device)
    actions = torch.tensor(batch.actions, device=device)
    old = torch.tensor(batch.log_probs, device=device)
    advantages, returns = duration_gae(batch.rewards, [0.] * (len(batch.rewards)+1) if supervised else batch.values,
                                      batch.durations, batch.dones, args.kappa, lam=1. if supervised else .95)
    advantages = torch.tensor(advantages, device=device)
    targets = torch.tensor(returns, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    # Optimizers are local and discarded each round; neither state is uploaded.
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    approx_kl = clip_fraction = 0.; early_stopped = False; completed_epochs = 0
    for _ in range(args.epochs):
        for ids in torch.randperm(len(states), device=device).split(128):
            logits = actor(states[ids])
            if not supervised:
                log_ratio = masked_distribution(logits, masks[ids]).log_prob(actions[ids]) - old[ids]
                approx_kl = float(((log_ratio.exp()-1)-log_ratio).mean().detach())
                clip_fraction = float(((log_ratio.exp()-1).abs() > .2).float().mean().detach())
                if approx_kl > args.target_kl:
                    early_stopped = True
                    break
            if supervised:
                # Regression on actually explored action returns. Unchosen actions
                # are never assigned invented counterfactual correctness targets.
                loss = nn.functional.smooth_l1_loss(logits.gather(1, actions[ids, None]).squeeze(1), targets[ids])
            else:
                loss, entropy = ppo_clip_loss(logits, masks[ids], actions[ids], old[ids], advantages[ids])
                loss = loss - .01 * entropy
            actor_opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), .5); actor_opt.step()
            value_loss = nn.functional.mse_loss(critic(states[ids]), targets[ids])
            critic_opt.zero_grad(); value_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), .5); critic_opt.step()
        completed_epochs += 1
        if early_stopped: break
    return {'approx_kl': approx_kl, 'clip_fraction': clip_fraction,
            'target_kl_early_stop': early_stopped, 'epochs_visited': completed_epochs,
            'actor_loss': float(loss.detach()), 'critic_loss': float(value_loss.detach()), 'transitions': len(states)}


def metrics(results, labels):
    classes = ['good', 'rust', 'corrosion', 'missing_cap', 'bird_nest', 'other']
    matrix = np.zeros((6, 7), dtype=int)
    ontime_matrix = np.zeros((6, 7), dtype=int)
    risk = 0.; covered = 0; ontime = 0
    recalls = {}
    latencies = []
    for record in results:
        truth = labels[record['item_id']]
        if isinstance(truth, dict): truth = truth['label']
        prediction = record.get('prediction')
        accepted = prediction in classes and not record.get('abstained', False)
        i = classes.index(truth); j = classes.index(prediction) if accepted else 6
        matrix[i, j] += 1
        timely = bool(record['within_deadline']); ontime += timely
        ontime_matrix[i, j if timely else 6] += 1
        covered += accepted
        if not accepted: risk += .5
        elif i != j: risk += 10 if i > 0 and j == 0 else 1
        latencies.append(record['elapsed_s'])
    def scores(cm):
        tp = int(cm[1:, 1:6].sum()); fp = int(cm[0, 1:6].sum()); fn = int(cm[1:, 0].sum()+cm[1:, 6].sum())
        binary = 2*tp / max(1, 2*tp+fp+fn)
        f1 = [2*cm[i,i] / max(1, cm[i,:].sum()+cm[:,i].sum()) for i in range(6)]
        return {'binary_defect_f1': binary, 'six_class_macro_f1': float(np.mean(f1))}
    for i, name in enumerate(classes): recalls[name] = float(matrix[i,i] / max(1, matrix[i,:].sum()))
    n = len(results)
    return {**scores(matrix), 'on_time': scores(ontime_matrix), 'items': n,
            'coverage': covered/max(n,1), 'deadline_success': ontime/max(n,1),
            'risk_per_1000': risk*1000/max(n,1), 'class_recall': recalls,
            'confusion_with_abstain': matrix.tolist(),
            'cache_replay_p50_s': float(np.median(latencies)) if n else None,
            'cache_replay_p95_s': float(np.percentile(latencies,95)) if n else None,
            'energy_per_verified_finding': None}


def evaluate(bundle, actors, budgets=(2.,6.,15.), split='val', device='cpu', enhance=False,
             max_eval_items=None, heuristic=False, network=None):
    if split not in ('val', 'test'): raise ValueError('Evaluation must be held-out')
    if len(actors) > 1:
        return {'split': split, 'local_evaluation_rule': 'each independent controller evaluated separately on the identical held-out missions',
                'controllers': {str(client): evaluate(bundle, {-1: actor}, budgets, split, device, enhance, max_eval_items, heuristic, network)
                                for client, actor in actors.items()}}
    items = [x for x in bundle.items if x['split'] == split]
    if max_eval_items is not None: items = items[:max_eval_items]
    output = {}
    for budget in budgets:
        results = []; traces = []; mission_latencies = []
        missions, excluded = evaluation_missions(items, budget)
        if excluded:
            raise ValueError(f'Budget {budget} unsupported: {len(excluded)} items cannot form same-tower distinct-source groups; no denominator exclusions allowed')
        for group in missions:
            first = group[0]
            client = int(first.get('client', -1))
            selected = [actors[client]] if client in actors else list(actors.values())
            env = InspectionEnv(bundle, group, budget_s=budget, feedback='audit', allow_enhance=enhance, network=network)
            state = env.reset(); done = False
            while not done:
                if device == 'cuda': torch.cuda.synchronize()
                start = time.perf_counter(); mask = np.asarray(env.mask(), dtype=bool)
                with torch.no_grad():
                    x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
                    logits = torch.stack([a(x) for a in selected]).mean(0)
                    action = heuristic_action(state, mask) if heuristic else int(logits.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf).argmax())
                if device == 'cuda': torch.cuda.synchronize()
                state, _, done, _ = env.step(action, policy_duration_s=time.perf_counter()-start)
            results.extend(env.results); traces.extend(env.trace); mission_latencies.append(env.elapsed_s)
        # Deliberately first access evaluator labels only after all missions finish.
        report = metrics(results, bundle.evaluation_labels)
        report['item_deadline_success'] = report['deadline_success']
        report['deadline_success'] = sum(t<=budget for t in mission_latencies)/max(1,len(mission_latencies))
        report.update(missions=len(mission_latencies),
                      excluded_ungroupable_item_ids=excluded, eligible_items=len(items)-len(excluded),
                      cache_replay_mission_p50_s=float(np.median(mission_latencies)) if mission_latencies else None,
                      cache_replay_mission_p95_s=float(np.percentile(mission_latencies,95)) if mission_latencies else None,
                      unique_source_images=len({x['source_id'] for group in missions for x in group}),
                      tool_calls=len(traces),
                      tool_action_counts={name: sum(t['tool']==name for t in traces) for name in sorted({t['tool'] for t in traces})},
                      cloud_bytes_up=sum(t.get('bytes_up',0) for t in traces if t['tool']=='CLOUD_QWEN8B'),
                      cloud_bytes_down=sum(t.get('bytes_down',0) for t in traces if t['tool']=='CLOUD_QWEN8B'),
                      cloud_byte_measurement_complete=all(t.get('bytes_measurement_valid',True) for t in traces if t['tool']=='CLOUD_QWEN8B'))
        output[str(budget)] = report
    return {'split': split, 'incomplete_smoke': max_eval_items is not None,
            'scope_extension_enhancement': enhance, 'network_config': network, 'latency_basis': 'cached tool duration plus measured controller execution; not target deployment latency',
            'budgets': output}


def export_actor(actor, path, metadata):
    layers = []
    for layer in actor.net:
        if isinstance(layer, nn.Linear):
            layers.append({'weight': layer.weight.detach().cpu().tolist(), 'bias': layer.bias.detach().cpu().tolist(), 'activation': 'linear'})
        elif isinstance(layer, nn.Tanh): layers[-1]['activation'] = 'tanh'
    document = {'schema': SCHEMA, 'features': list(FEATURES), 'actions': list(ACTIONS),
                'input_transform': {'name': 'signed_log1p', 'clip': 1e6, 'nonfinite': 0},
                'layers': layers, 'metadata': metadata}
    path.write_text(json.dumps(document))


def save_checkpoint(path, actors, metadata):
    torch.save({'schema': SCHEMA, 'features': list(FEATURES), 'actions': list(ACTIONS),
                'actors': {str(k): {name: t.detach().cpu() for name,t in a.state_dict().items()} for k,a in actors.items()},
                'metadata': metadata}, path)


def train(args):
    args.network = json.loads(Path(args.network_config).read_text()) if args.network_config else None
    if args.network is not None:
        if not isinstance(args.network, dict) or set(args.network) - {'available','rtt_s','bandwidth_mbps','delay_s'}:
            raise ValueError('Network config must contain only available/rtt_s/bandwidth_mbps/delay_s')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    bundle = Bundle(args.bundle, feedback=args.feedback)
    if bundle.manifest.get('synthetic_fixture') and not args.allow_synthetic:
        raise ValueError('Synthetic fixtures require --allow-synthetic; never scientific results')
    complete = set(bundle.manifest.get('cache_complete_for_splits', []))
    required = {args.eval_split} if args.evaluate_only else {'train', args.eval_split}
    if not bundle.manifest.get('cache_complete') and not required <= complete:
        raise ValueError(f'Incomplete replay cache for {required}')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    if not args.evaluate_only and args.mode != 'heuristic' and args.device != 'cuda':
        raise ValueError('Training is CUDA-only by user requirement')
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.evaluate_only:
        saved = torch.load(args.evaluate_only, map_location='cpu', weights_only=True)
        if saved['schema'] != SCHEMA or saved['features'] != list(FEATURES) or saved['actions'] != list(ACTIONS):
            raise ValueError('Checkpoint interface mismatch')
        saved_metadata = saved['metadata']
        if bool(saved_metadata.get('scope_extension_enhancement')) != args.enhance:
            raise ValueError('Checkpoint enhancement scope differs from requested evaluation')
        signature = bundle.manifest.get('provenance', {}).get('tool_cache', {}).get('signature')
        if saved_metadata.get('cache_signature') != signature:
            raise ValueError('Checkpoint frozen-tool cache signature differs from evaluation bundle')
        actors = {}
        for key,state in saved['actors'].items():
            actor = Actor().to(args.device); actor.load_state_dict(state); actors[int(key)] = actor
        result = evaluate(bundle, actors, args.budgets, args.eval_split, args.device, args.enhance, args.max_eval_items,
                          heuristic=saved['metadata'].get('mode') == 'heuristic', network=args.network)
        (output/'evaluation.json').write_text(json.dumps(result, indent=2)); return result
    training = [x for x in bundle.items if x['split']=='train']
    clients = sorted({int(x['client']) for x in training})
    if not training or clients != list(range(5)): raise ValueError('Expected all five training clients')
    global_actor = Actor().to(args.device)
    initial = {k:v.detach().clone() for k,v in global_actor.state_dict().items()}
    actors = {}; critics = {}
    for client in clients:
        actors[client] = Actor().to(args.device); actors[client].load_state_dict(initial)
        critics[client] = Critic().to(args.device)
    cumulative = 0; logs = []; started = time.perf_counter()
    for round_id in range(args.rounds):
        version = round_id
        reference = {k:v.detach().cpu().clone() for k,v in global_actor.state_dict().items()}
        deltas=[]; payloads=[]; weights=[]; round_logs=[]
        groups = [(-1,training)] if args.mode in ('central', 'supervised') else [(c,[x for x in training if int(x['client'])==c]) for c in clients]
        for client,pool in groups:
            key = 0 if client == -1 else client
            actor = actors[key]; critic = critics[key]
            if args.mode == 'fed': actor.load_state_dict(reference)
            if args.mode in ('central', 'supervised'):
                batches, results = [], []
                # Hold this shared actor fixed across all five regional draws;
                # use precisely the same client seeds/counts as federation.
                for region in clients:
                    region_pool = [x for x in training if int(x['client']) == region]
                    part, inspected = collect(bundle, region_pool, actor, critic, args, version,
                                              args.seed+round_id*10000+region*1000,
                                              args.episodes_per_client,
                                              behavior='supervised' if args.mode=='supervised' else None)
                    batches.append(part); results.extend(inspected)
                batch = merge_rollouts(batches, version)
            else:
                batch, results = collect(bundle,pool,actor,critic,args,version,args.seed+round_id*10000+key*1000,
                                         args.episodes_per_client,
                                         behavior=args.mode if args.mode=='heuristic' else None)
            # Persist private rollout evidence locally; these files are NEVER
            # included in the actor-delta payload or federation byte total.
            torch.save({'schema': 'fedorch-ppo-v2-rollout-v1', 'policy_version': batch.version,
                        'states': torch.as_tensor(np.asarray(batch.states)),
                        'masks': torch.as_tensor(np.asarray(batch.masks)),
                        'actions': torch.tensor(batch.actions), 'old_log_probs': torch.tensor(batch.log_probs),
                        'rewards': torch.tensor(batch.rewards), 'durations': torch.tensor(batch.durations),
                        'dones': torch.tensor(batch.dones), 'values': torch.tensor(batch.values),
                        'decision_records': batch.records},
                       output/f'private-rollout-r{round_id:02d}-c{client}.pt')
            details = {} if args.mode == 'heuristic' else update(actor,critic,batch,version,args,supervised=args.mode=='supervised')
            if args.mode != 'heuristic':
                torch.save({k: v.detach().cpu() for k,v in critic.state_dict().items()},
                           output/f'private-critic-r{round_id:02d}-c{client}.pt')
            details.update(client=client, inspected_items=len(results), audited_items=sum(bool(x.get('audited')) for x in results),
                           audited_unique_items=len({x['item_id'] for x in results if x.get('audited')}))
            round_logs.append(details)
            if args.mode == 'fed':
                payload,delta = serialize_delta(actor,reference)
                payloads.append(payload); deltas.append(delta)
                weights.append(len(batch.states) if args.fed_weighting=='sample' else 1)
        if args.mode == 'fed':
            try: total = preflight_payloads(payloads,cumulative)
            except ValueError as error:
                logs.append({'round':round_id,'stopped_before_upload':str(error)}); break
            for client,payload in zip(clients,payloads): (output/f'delta-r{round_id:02d}-c{client}.pt').write_bytes(payload)
            cumulative=total; global_actor.load_state_dict(fedavg(reference,deltas,weights))
            current={-1:global_actor}
        elif args.mode in ('central', 'supervised'): current={-1:actors[0]}
        elif args.mode == 'heuristic': current={-1:actors[0]}
        else: current=actors
        metadata={'cache_signature':bundle.manifest.get('provenance', {}).get('tool_cache', {}).get('signature'),
                  'client_role':'drone', 'local_critic_owner':'drone',
                  'federation_role':'actor-only coordinator; no images or trajectories uploaded',
                  'network_config':args.network, 'mode':args.mode,'feedback':args.feedback,'seed':args.seed,'round':round_id,
                  'serialized_actor_payload_bytes':cumulative,'transport':'serialized replay payload, not measured network wire bytes',
                  'scope_extension_enhancement':args.enhance,'synthetic_fixture':bool(bundle.manifest.get('synthetic_fixture')),
                  'supervised_method':'observed-action discounted return regression, epsilon=0.2; audit-IPW error only; no counterfactual oracle',
                  'delta_payload_files': [f'delta-r{round_id:02d}-c{c}.pt' for c in clients] if args.mode=='fed' else [],
                  'actor_parameters':sum(p.numel() for p in global_actor.parameters()),'training_split':'train',
                  'audit_unique_available':sum(bool(v['audited']) for v in bundle.audits.values()),
                  'training_seconds':time.perf_counter()-started,'rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        checkpoint=output/f'round-{round_id:02d}.pt'; save_checkpoint(checkpoint,current,metadata)
        for key,actor in current.items(): export_actor(actor,output/f'actor-r{round_id:02d}-c{key}.json',metadata)
        # Test evaluation is strictly once, after the final fixed policy.
        evaluation = None if args.eval_split=='test' else evaluate(bundle,current,args.budgets,'val',args.device,args.enhance,args.max_eval_items,args.mode=='heuristic',args.network)
        record={'round':round_id,'clients':round_logs,'metadata':metadata,'evaluation':evaluation}
        logs.append(record); (output/'summary.json').write_text(json.dumps(logs,indent=2))
        print(json.dumps(record),flush=True)
    if args.eval_split=='test' and logs and 'metadata' in logs[-1]:
        (output/'final-test.json').write_text(json.dumps(evaluate(bundle,current,args.budgets,'test',args.device,args.enhance,None,args.mode=='heuristic',args.network),indent=2))
    (output/'summary.json').write_text(json.dumps(logs,indent=2))
    return {'rounds':logs,'serialized_actor_payload_bytes':cumulative}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True); p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['fed','central','local','supervised','heuristic'],default='fed')
    p.add_argument('--feedback',choices=['audit','full'],default='audit')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--episodes-per-client',type=int,default=64)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--budgets',type=lambda s:[float(x) for x in s.split(',')],default=[2.,6.,15.])
    p.add_argument('--eval-split',choices=['val','test'],default='val')
    p.add_argument('--max-eval-items',type=int)
    p.add_argument('--network-config',metavar='JSON_FILE', help='Explicit measured network scenario; default disconnected/cloud masked')
    p.add_argument('--enhance',action='store_true'); p.add_argument('--allow-synthetic',action='store_true')
    p.add_argument('--evaluate-only',metavar='CHECKPOINT')
    p.add_argument('--fed-weighting',choices=['sample','equal'],default='sample')
    p.add_argument('--epochs',type=int,default=4); p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--kappa',type=float,default=.01)
    p.add_argument('--target-kl',type=float,default=.03)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.target_kl<=0 or args.epochs<1 or args.rounds<1 or args.episodes_per_client<1 or not args.budgets or any(x<=0 for x in args.budgets):
        raise ValueError('Positive rounds, episode count and budgets required')
    train(args)


if __name__=='__main__': main()
