"""Flower/Ray drone simulation: full actor downlink, FP16 actor-delta uplink.

Strategies: Flower FedAvg and FedProx aggregate actor deltas weighted by
transitions (or equally). Optional server optimizers act on the averaged delta:
momentum gives FedAvgM (Hsu et al. 2019), adam gives FedAdam (Reddi et al. 2021).
fraction_fit samples a subset of drones per round (partial participation).

Critics, optimizer state, replay observations and trajectories remain in each
simulated drone's local directory. Protobuf counts measure application payloads,
not Ray envelopes, TCP packets, or physical drone network traffic.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import (Code, FitRes, Status, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.common.serde import fit_ins_to_proto, fit_res_to_proto, parameters_to_proto
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

FLEET_SIZE = 5
CAP_BYTES = 30_000_000
RESPONSE_ALLOWANCE = 64 * 1024


def encode_actor(state):
    """Preserve state-dict insertion order; never include critic/optimizer state."""
    return [value.detach().cpu().numpy().astype(np.float32, copy=True)
            for value in state.values()]


def validate_arrays(arrays, reference, dtype):
    if len(arrays) != len(reference):
        raise ValueError('Actor tensor count mismatch')
    for array, expected in zip(arrays, reference):
        if array.shape != expected.shape or array.dtype != np.dtype(dtype):
            raise ValueError('Actor tensor shape/dtype mismatch')
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite actor payload')


def decode_actor(arrays, state):
    validate_arrays(arrays, encode_actor(state), np.float32)
    return {key: torch.from_numpy(array.copy()) for key, array in zip(state, arrays)}


def encode_delta(state, reference):
    arrays = encode_actor(state)
    validate_arrays(reference, arrays, np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        deltas = [(array - anchor).astype(np.float16) for array, anchor in zip(arrays, reference)]
    validate_arrays(deltas, reference, np.float16)
    return deltas


def response_bytes(arrays, count, metrics):
    response = FitRes(Status(Code.OK, ''), ndarrays_to_parameters(arrays), count, metrics)
    return fit_res_to_proto(response).ByteSize()


def response_bound(reference):
    """Exact serialized FP16 Parameters plus bounded FitRes metadata allowance."""
    zeros = [np.zeros_like(array, dtype=np.float16) for array in reference]
    return parameters_to_proto(ndarrays_to_parameters(zeros)).ByteSize() + RESPONSE_ALLOWANCE


class ActorDeltaMixin:
    """Adapt Flower's real weighted strategies to actor delta aggregation."""

    def __init__(self, reference, *, weighting='sample', byte_cap=CAP_BYTES,
                 round_callback=None, fraction_fit=1., server_opt='none', server_lr=None,
                 server_momentum=.9, **kwargs):
        if weighting not in ('sample', 'equal'):
            raise ValueError('Unknown federation weighting')
        if server_opt not in ('none', 'momentum', 'adam'):
            raise ValueError('Unknown server optimizer')
        self.sampled = max(1, math.ceil(fraction_fit * FLEET_SIZE))
        self.server_opt = server_opt
        self.server_lr = server_lr if server_lr is not None else (1e-3 if server_opt == 'adam' else 1.)
        self.server_momentum = server_momentum
        self.first_moment = None
        self.second_moment = None
        self.reference = [np.asarray(x, dtype=np.float32).copy() for x in reference]
        validate_arrays(self.reference, self.reference, np.float32)
        self.weighting = weighting
        self.byte_cap = byte_cap
        self.response_limit = response_bound(self.reference)
        self.uplink_bytes = 0
        self.downlink_bytes = 0
        self.history = []
        self.stop_reason = None
        self.failure = None
        self.round_callback = round_callback
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=0.,
                         min_fit_clients=self.sampled, min_evaluate_clients=0,
                         min_available_clients=FLEET_SIZE, accept_failures=False,
                         initial_parameters=ndarrays_to_parameters(self.reference),
                         fit_metrics_aggregation_fn=lambda _: {}, **kwargs)
        # FedProx 1.43 has no `inplace` constructor argument. Its inherited
        # aggregate_fit still supports disabling in-place arithmetic explicitly.
        self.inplace = False

    def configure_fit(self, server_round, parameters, client_manager):
        if self.stop_reason:
            return []
        reserved = self.sampled * self.response_limit
        if self.uplink_bytes + reserved >= self.byte_cap:
            self.stop_reason = (f'Round {server_round} stopped before dispatch: '
                                f'{self.uplink_bytes}+{reserved} would reach strict <{self.byte_cap} cap')
            return []
        validate_arrays(parameters_to_ndarrays(parameters), self.reference, np.float32)
        pairs = super().configure_fit(server_round, parameters, client_manager)
        if len(pairs) != self.sampled:
            raise RuntimeError(f'Expected {self.sampled} sampled drones, got {len(pairs)}')
        for _, instruction in pairs:
            instruction.config.update(policy_version=server_round - 1,
                                      response_limit=self.response_limit,
                                      fed_weighting=self.weighting)
        self.downlink_bytes += sum(fit_ins_to_proto(ins).ByteSize() for _, ins in pairs)
        return pairs

    def aggregate_fit(self, server_round, results, failures):
        try:
            sizes = [fit_res_to_proto(result).ByteSize() for _, result in results]
            self.uplink_bytes += sum(sizes)
            if failures or len(results) != self.sampled:
                raise RuntimeError('Incomplete Flower round; partial aggregation forbidden')
            if self.uplink_bytes >= self.byte_cap or any(size > self.response_limit for size in sizes):
                raise RuntimeError('Flower response exceeded reserved strict payload budget')
            decoded = []
            ids = set()
            client_metrics = []
            for proxy, result in results:
                metric = result.metrics
                drone = metric.get('client')
                if (result.status.code != Code.OK or type(drone) is not int or
                        drone not in range(FLEET_SIZE) or drone in ids):
                    raise ValueError('Invalid/duplicate drone response')
                ids.add(drone)
                if metric.get('policy_version') != server_round - 1:
                    raise ValueError('Stale policy version')
                actual = metric.get('transitions')
                if type(actual) is not int or actual <= 0:
                    raise ValueError('Missing actual transition count')
                expected = actual if self.weighting == 'sample' else 1
                if result.num_examples != expected:
                    raise ValueError('Incorrect federation weight')
                arrays = parameters_to_ndarrays(result.parameters)
                validate_arrays(arrays, self.reference, np.float16)
                # Flower must accumulate FP32 copies, not low precision tensors.
                decoded.append((proxy, replace(result, parameters=ndarrays_to_parameters(
                    [array.astype(np.float32) for array in arrays]))))
                client_metrics.append(dict(metric))
            average, metrics = super().aggregate_fit(server_round, decoded, [])
            if average is None:
                raise RuntimeError('Flower produced no aggregated delta')
            deltas = self.server_step(parameters_to_ndarrays(average))
            self.reference = [anchor + delta for anchor, delta in zip(self.reference, deltas)]
            validate_arrays(self.reference, self.reference, np.float32)
            record = {'round': server_round - 1, 'clients': sorted(client_metrics, key=lambda m: m['client']),
                      'participants': sorted(m['client'] for m in client_metrics),
                      'server_opt': self.server_opt, 'server_lr': self.server_lr,
                      'round_flower_fitres_protobuf_bytes': sum(sizes),
                      'flower_fitres_protobuf_bytes': self.uplink_bytes,
                      'flower_fitins_protobuf_downlink_bytes': self.downlink_bytes}
            self.history.append(record)
            if self.round_callback:
                self.round_callback(server_round, self.reference, record)
            return ndarrays_to_parameters(self.reference), metrics
        except Exception as error:
            self.failure = str(error)
            raise


def _server_step(self, delta):
    """FedAvg: theta += delta. FedAvgM: m = beta*m + delta. FedAdam: Adam moments, tau 1e-3."""
    if self.server_opt == 'none':
        return [self.server_lr * d for d in delta]
    if self.server_opt == 'momentum':
        if self.first_moment is None:
            self.first_moment = [np.zeros_like(d) for d in delta]
        self.first_moment = [self.server_momentum * m + d for m, d in zip(self.first_moment, delta)]
        return [self.server_lr * m for m in self.first_moment]
    if self.first_moment is None:
        self.first_moment = [np.zeros_like(d) for d in delta]
        self.second_moment = [np.zeros_like(d) for d in delta]
    self.first_moment = [.9 * m + .1 * d for m, d in zip(self.first_moment, delta)]
    self.second_moment = [.99 * v + .01 * d * d for v, d in zip(self.second_moment, delta)]
    return [self.server_lr * m / (np.sqrt(v) + 1e-3) for m, v in zip(self.first_moment, self.second_moment)]


ActorDeltaMixin.server_step = _server_step


class ActorDeltaFedAvg(ActorDeltaMixin, FedAvg):
    pass


class ActorDeltaFedProx(ActorDeltaMixin, FedProx):
    pass


def private_directory(output, drone):
    if type(drone) is not int or drone not in range(FLEET_SIZE):
        raise ValueError('Expected drone partition 0..4')
    directory = Path(output) / f'drone-{drone}'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_private_critic(critic, directory):
    path = directory / 'privatecritic.pt'
    if path.exists():
        critic.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))


def save_private_critic(critic, directory):
    torch.save({key: value.detach().cpu() for key, value in critic.state_dict().items()},
               directory / 'privatecritic.pt')


def save_private_rollout(batch, directory, version):
    torch.save({'schema': 'fedorch-ppo-v2-rollout-v1', 'policy_version': batch.version,
                'states': torch.as_tensor(np.asarray(batch.states)),
                'masks': torch.as_tensor(np.asarray(batch.masks)),
                'actions': torch.tensor(batch.actions), 'old_log_probs': torch.tensor(batch.log_probs),
                'rewards': torch.tensor(batch.rewards), 'durations': torch.tensor(batch.durations),
                'dones': torch.tensor(batch.dones), 'values': torch.tensor(batch.values),
                'decision_records': batch.records}, directory / f'private-rollout-r{version:02d}.pt')


class DroneClient(NumPyClient):
    def __init__(self, drone, args, bundle):
        self.drone, self.args, self.bundle = drone, args, bundle

    def fit(self, parameters, config):
        # Lazy import prevents a cycle with train()'s Flower dispatch.
        from .ppo_v2_train import Actor, Critic, collect, update, cuda_or_synthetic
        args = self.args
        cuda_or_synthetic(args)
        version = int(config['policy_version'])
        seed = args.seed + version * 10000 + self.drone * 1000
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        pool = [item for item in self.bundle.items
                if item['split'] == 'train' and int(item['client']) == self.drone]
        if not pool:
            raise ValueError('Drone training partition is empty')
        directory = private_directory(args.output, self.drone)
        actor = Actor().to(args.device)
        actor.load_state_dict(decode_actor(parameters, actor.state_dict()))
        critic = Critic().to(args.device)
        load_private_critic(critic, directory)
        anchor = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        try:
            batch, results = collect(self.bundle, pool, actor, critic, args, version, seed,
                                     args.episodes_per_client)
            if batch.version != version or not batch.states:
                raise ValueError('Fresh nonempty on-policy rollout required')
            save_private_rollout(batch, directory, version)
            details = update(actor, critic, batch, version, args,
                             proximal_mu=float(config.get('proximal_mu', 0.)), anchor=anchor)
            save_private_critic(critic, directory)
            deltas = encode_delta(actor.state_dict(), parameters)
            metrics = {'client': self.drone, 'policy_version': version,
                       'transitions': len(batch.states), 'inspected_items': len(results),
                       'audited_items': sum(bool(x.get('audited')) for x in results),
                       'audited_unique_items': len({x['item_id'] for x in results if x.get('audited')})}
            for key in ('approx_kl', 'clip_fraction', 'target_kl_early_stop', 'epochs_visited',
                        'actor_loss', 'critic_loss'):
                if key in details:
                    value = details[key]
                    metrics[key] = bool(value) if isinstance(value, bool) else float(value)
            count = len(batch.states) if config['fed_weighting'] == 'sample' else 1
            # Enforce the entire FitRes bound before NumPyClient transmits it.
            if response_bytes(deltas, count, metrics) > int(config['response_limit']):
                raise ValueError('Client response exceeds reserved Flower message budget')
            return deltas, count, metrics
        finally:
            del actor, critic, anchor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DroneFactory:
    """Picklable client_fn: Flower requires the exact signature (context)."""

    def __init__(self, args, bundle):
        self.args, self.bundle = args, bundle

    def __call__(self, context):
        return DroneClient(int(context.node_config['partition-id']), self.args, self.bundle).to_client()


class ServerFactory:
    def __init__(self, strategy, rounds):
        self.strategy, self.rounds = strategy, rounds

    def __call__(self, context):
        return ServerAppComponents(strategy=self.strategy, config=ServerConfig(num_rounds=self.rounds))


def run_flower(args, bundle):
    """Run five virtual drones sequentially on one CUDA GPU through real Flower."""
    from flwr.simulation import run_simulation
    from .ppo_v2_train import Actor, evaluate, export_actor, save_checkpoint, cuda_or_synthetic
    cuda_or_synthetic(args)
    clients = {int(x['client']) for x in bundle.items if x['split'] == 'train'}
    if clients != set(range(FLEET_SIZE)):
        raise ValueError('Expected all five drone training partitions')
    output = Path(args.output).resolve()
    args.output = str(output)
    output.mkdir(parents=True, exist_ok=True)
    # Construct the same seeded initial actor as the central baseline. Driver
    # retains only CPU weights between rounds; all learning is CUDA worker-side.
    global_actor = Actor()
    reference = encode_actor(global_actor.state_dict())
    started = time.perf_counter()
    strategy_name = args.flower_strategy
    if strategy_name not in ('fedavg', 'fedprox'):
        raise ValueError('Unknown Flower strategy')

    def evaluate_fixed(split, max_items):
        global_actor.to(args.device)
        try:
            return evaluate(bundle, {-1: global_actor}, args.budgets, split, args.device,
                            args.enhance, max_items, network=args.network)
        finally:
            global_actor.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def after_round(server_round, arrays, record):
        global_actor.load_state_dict(decode_actor(arrays, global_actor.state_dict()))
        version = server_round - 1
        metadata = {'cache_signature': bundle.manifest.get('provenance', {}).get('tool_cache', {}).get('signature'),
                    'client_role': 'drone', 'local_critic_owner': 'drone',
                    'federation_role': 'actor-only coordinator; no images or trajectories uploaded',
                    'mode': 'fed', 'flower_strategy': strategy_name,
                    'proximal_mu': args.proximal_mu if strategy_name == 'fedprox' else 0.,
                    'server_opt': args.server_opt, 'server_lr': strategy.server_lr,
                    'fraction_fit': args.fraction_fit, 'drones_per_round': strategy.sampled,
                    'fed_weighting': args.fed_weighting, 'feedback': args.feedback,
                    'seed': args.seed, 'round': version, 'network_config': args.network,
                    'scope_extension_enhancement': args.enhance,
                    'synthetic_fixture': bool(bundle.manifest.get('synthetic_fixture')),
                    'training_split': 'train', 'simulation_backend': 'Flower Ray',
                    'actor_parameters': sum(p.numel() for p in global_actor.parameters()),
                    'serialized_actor_payload_bytes': strategy.uplink_bytes,
                    'flower_fitres_protobuf_bytes': strategy.uplink_bytes,
                    'flower_fitins_protobuf_downlink_bytes': strategy.downlink_bytes,
                    'transport': 'Flower FitRes protobuf application bytes, not measured network wire bytes',
                    'audit_unique_available': sum(bool(v['audited']) for v in bundle.audits.values()),
                    'training_seconds': time.perf_counter() - started}
        save_checkpoint(output / f'round-{version:02d}.pt', {-1: global_actor}, metadata)
        export_actor(global_actor, output / f'actor-r{version:02d}-c-1.json', metadata)
        record['metadata'] = metadata
        record['evaluation'] = evaluate_fixed('val', args.max_eval_items) if args.eval_split == 'val' else None
        (output / 'summary.json').write_text(json.dumps(strategy.history, indent=2))
        print(json.dumps(record), flush=True)

    strategy_type = ActorDeltaFedAvg if strategy_name == 'fedavg' else ActorDeltaFedProx
    extra = {'proximal_mu': args.proximal_mu} if strategy_name == 'fedprox' else {}
    strategy = strategy_type(reference, weighting=args.fed_weighting, round_callback=after_round,
                             fraction_fit=args.fraction_fit, server_opt=args.server_opt,
                             server_lr=args.server_lr, server_momentum=args.server_momentum, **extra)
    gpus = 1 if args.device == 'cuda' else 0
    run_simulation(
        server_app=ServerApp(server_fn=ServerFactory(strategy, args.rounds)),
        client_app=ClientApp(client_fn=DroneFactory(args, bundle)),
        num_supernodes=FLEET_SIZE, backend_name='ray',
        backend_config={'client_resources': {'num_cpus': 1, 'num_gpus': gpus},
                        'init_args': {'num_cpus': 2, 'num_gpus': gpus,
                                      'object_store_memory': 256 * 1024 * 1024,
                                      'include_dashboard': False}})
    if strategy.failure:
        raise RuntimeError(f'Flower round failed: {strategy.failure}')
    if args.eval_split == 'test' and strategy.history:
        (output / 'final-test.json').write_text(json.dumps(evaluate_fixed('test', None), indent=2))
    (output / 'summary.json').write_text(json.dumps(strategy.history, indent=2))
    summary = {'rounds': strategy.history,
               'flower_strategy': strategy_name, 'server_opt': args.server_opt,
               'fraction_fit': args.fraction_fit,
               'serialized_actor_payload_bytes': strategy.uplink_bytes,
               'flower_fitres_protobuf_bytes': strategy.uplink_bytes,
               'flower_fitins_protobuf_downlink_bytes': strategy.downlink_bytes,
               'stop_reason': strategy.stop_reason}
    (output / 'flower-summary.json').write_text(json.dumps(summary, indent=2))
    return summary
