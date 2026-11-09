"""Base laya as the agent: PPO on laya's scorer over cached option vectors.

One episode inspects one component image under a time budget. In each state, base laya scores
the next action (write the report, zoom into one of five crops, run ResNet on the view, ask Qwen)
and, when it writes the report, the three report questions. The trainable part is laya's option
scorer (shared by all questions, initialised from base laya) plus a small state term for the
action logits. The backbone and head transformer are frozen; their outputs are cached.
"""
import json
import math

import numpy as np
import torch
from torch import nn

ACTIONS = 8
SIZES = (8, 11, 2, 6)
OFFSETS = np.cumsum((0,) + SIZES)
REPORT, ZOOMS, RESNET, QWEN = 0, range(1, 6), 6, 7
STATE = 12
MISS, FALSE_ALARM, WRONG = 10., 1., 1.


class Cache:
    def __init__(self, root, splits=("val", "test")):
        self.meta = [m for m in map(json.loads, open(root / "meta.jsonl")) if m["split"] in splits]
        self.vectors = [np.load(root / m["file"], mmap_mode="r") for m in self.meta]
        self.scorer = torch.load(root / "base_scorer.pt", map_location="cpu")


class Policy(nn.Module):
    def __init__(self, scorer_state):
        super().__init__()
        d = 576
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.scorer.load_state_dict(scorer_state)
        self.state = nn.Sequential(nn.Linear(STATE, 32), nn.Tanh(), nn.Linear(32, ACTIONS))
        nn.init.zeros_(self.state[2].weight)
        nn.init.zeros_(self.state[2].bias)

    def logits(self, vectors, state):
        raw = self.scorer(vectors.float()).squeeze(-1)
        action = raw[..., OFFSETS[0]:OFFSETS[1]] + self.state(state)
        return action, [raw[..., OFFSETS[q]:OFFSETS[q + 1]] for q in (1, 2, 3)]


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(576 + STATE, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, vectors, state):
        return self.net(torch.cat([vectors[..., :ACTIONS, :].float().mean(-2), state], -1)).squeeze(-1)


class Episode:
    """One image, one budget. State: current view, ResNet shown on this view, Qwen shown, elapsed time."""

    def __init__(self, cache, index, budget, network_s=0., tools=True):
        self.m, self.vec, self.budget, self.network_s, self.tools = cache.meta[index], cache.vectors[index], budget, network_s, tools
        self.view, self.resnet, self.qwen, self.elapsed, self.steps = 0, 0, 0, 0., 0

    def vectors(self):
        return torch.from_numpy(np.asarray(self.vec[self.view, self.resnet, self.qwen]))

    def decide_s(self):
        return self.m["laya_s"][self.view][self.resnet][self.qwen]

    def features(self):
        x = np.zeros(STATE, np.float32)
        x[0], x[1] = self.elapsed / self.budget, max(0., self.budget - self.elapsed) / self.budget
        x[2], x[3 + self.view] = math.log(self.budget), 1.
        x[9], x[10], x[11] = self.resnet, self.qwen, self.steps / 6
        return torch.from_numpy(x)

    def cost(self, action):
        if action in ZOOMS:
            return self.m["crop_s"][action - 0]
        if action == RESNET:
            return self.m["resnet_s"][self.view]
        if action == QWEN:
            return self.m["qwen_s"] + self.network_s
        return 0.

    def mask(self):
        mask = np.zeros(ACTIONS, bool)
        mask[REPORT] = True
        if not self.tools or self.steps >= 6:
            return torch.from_numpy(mask)
        left = self.budget - self.elapsed - self.decide_s()
        for a in range(1, ACTIONS):
            fits = self.cost(a) + 0.15 <= left
            ok = (a in ZOOMS and a != self.view) or (a == RESNET and not self.resnet) or (a == QWEN and not self.qwen)
            mask[a] = fits and ok
        return torch.from_numpy(mask)

    def step(self, action):
        duration = self.decide_s() + self.cost(action)
        self.elapsed += duration
        self.steps += 1
        if action in ZOOMS:
            self.view, self.resnet = action, 0
        elif action == RESNET:
            self.resnet = 1
        elif action == QWEN:
            self.qwen = 1
        return duration


def report_cost(meta, component, defect, condition):
    truth_defect = meta["defect"]
    cost = MISS if truth_defect and not defect else FALSE_ALARM if defect and not truth_defect else 0.
    return cost + WRONG * (condition != meta["condition"]) + WRONG * (component != meta["component"])


def rollout(policy, critic, cache, indices, budgets, rng, sample=True, tools=True, lambda_time=1., device="cpu"):
    batch, results = [], []
    for index in indices:
        budget = budgets[rng.integers(len(budgets))] if sample else budgets
        ep = Episode(cache, index, budget, tools=tools)
        steps = []
        while True:
            v, x, mask = ep.vectors().to(device), ep.features().to(device), ep.mask().to(device)
            with torch.no_grad():
                action_logits, report_logits = policy.logits(v, x)
                dist = torch.distributions.Categorical(logits=action_logits.masked_fill(~mask, -1e9))
                action = int(dist.sample()) if sample else int(dist.probs.argmax())
                logp = float(dist.log_prob(torch.tensor(action, device=device)))
                value = float(critic(v, x))
                answers = []
                if action == REPORT:
                    for logits in report_logits:
                        d = torch.distributions.Categorical(logits=logits)
                        choice = int(d.sample()) if sample else int(logits.argmax())
                        answers.append(choice)
                        logp += float(d.log_prob(torch.tensor(choice, device=device)))
            duration = ep.step(action)
            reward = -lambda_time * duration / ep.budget
            if action == REPORT:
                cost = report_cost(ep.m, answers[0], answers[1], answers[2])
                reward -= cost
                results.append({"image": ep.m["image"], "budget": ep.budget, "elapsed": ep.elapsed, "steps": ep.steps,
                                "component": answers[0], "defect": answers[1], "condition": answers[2], "cost": cost,
                                "truth": [ep.m["component"], ep.m["defect"], ep.m["condition"]]})
            steps.append(dict(v=v, x=x, mask=mask, action=action, answers=answers, logp=logp, value=value,
                              reward=reward, duration=duration, done=action == REPORT))
            if action == REPORT:
                break
        batch.append(steps)
    return batch, results


def advantages(episode, kappa=.5, lam=.95):
    running, out = 0., []
    for n in reversed(range(len(episode))):
        s = episode[n]
        gamma = math.exp(-kappa * s["duration"])
        next_value = 0. if s["done"] else episode[n + 1]["value"]
        delta = s["reward"] + gamma * next_value - s["value"]
        running = delta + gamma * lam * (0. if s["done"] else running)
        out.append(running)
    return out[::-1]


def update(policy, critic, batch, epochs=4, lr=3e-4, clip=.2, entropy=.01, prox=0., anchor=None):
    steps = [s for ep in batch for s in ep]
    adv = torch.tensor([a for ep in batch for a in advantages(ep)])
    returns = adv + torch.tensor([s["value"] for s in steps])
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    opt_p = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    v = torch.stack([s["v"] for s in steps])
    x = torch.stack([s["x"] for s in steps])
    mask = torch.stack([s["mask"] for s in steps])
    actions = torch.tensor([s["action"] for s in steps], device=v.device)
    old = torch.tensor([s["logp"] for s in steps], device=v.device)
    adv, returns = adv.to(v.device), returns.to(v.device)
    for _ in range(epochs):
        for ids in torch.randperm(len(steps)).split(256):
            action_logits, report_logits = policy.logits(v[ids], x[ids])
            dist = torch.distributions.Categorical(logits=action_logits.masked_fill(~mask[ids], -1e9))
            logp = dist.log_prob(actions[ids])
            ent = dist.entropy()
            for q, logits in enumerate(report_logits):
                d = torch.distributions.Categorical(logits=logits)
                chosen = torch.tensor([steps[i]["answers"][q] if steps[i]["answers"] else 0 for i in ids.tolist()], device=v.device)
                is_report = (actions[ids] == REPORT).float()
                logp = logp + is_report * d.log_prob(chosen)
                ent = ent + is_report * d.entropy()
            ratio = (logp - old[ids]).exp()
            loss = -torch.minimum(ratio * adv[ids], ratio.clamp(1 - clip, 1 + clip) * adv[ids]).mean() - entropy * ent.mean()
            if prox:
                loss = loss + prox / 2 * sum(((p - a) ** 2).sum() for p, a in zip(policy.parameters(), anchor))
            opt_p.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), .5); opt_p.step()
            value_loss = ((critic(v[ids], x[ids]) - returns[ids]) ** 2).mean()
            opt_c.zero_grad(); value_loss.backward(); opt_c.step()
    return {"policy_loss": float(loss), "value_loss": float(value_loss), "transitions": len(steps)}


def evaluate(policy, cache, indices, budgets, tools=True, device="cpu"):
    out = {}
    critic = Critic().to(device)
    for budget in budgets:
        _, results = rollout(policy, critic, cache, indices, budget, np.random.default_rng(0), sample=False, tools=tools, device=device)
        truth = np.array([r["truth"] for r in results])
        pred = np.array([[r["component"], r["defect"], r["condition"]] for r in results])
        defects = truth[:, 1] == 1
        miss = np.sum(defects & (pred[:, 1] == 0))
        false_alarm = np.sum(~defects & (pred[:, 1] == 1))
        out[str(budget)] = {"images": len(results),
                            "report_cost_per_1000": round(1000 * float(np.mean([r["cost"] for r in results])), 1),
                            "defect_risk_per_1000": round(1000 * (MISS * miss + FALSE_ALARM * false_alarm) / len(results), 1),
                            "defect_recall": round(float(np.mean(pred[defects, 1] == 1)), 4),
                            "false_alarm": round(float(np.mean(pred[~defects, 1] == 1)), 4),
                            "component_acc": round(float(np.mean(pred[:, 0] == truth[:, 0])), 4),
                            "condition_acc": round(float(np.mean(pred[:, 2] == truth[:, 2])), 4),
                            "mean_steps": round(float(np.mean([r["steps"] for r in results])), 2),
                            "mean_time_s": round(float(np.mean([r["elapsed"] for r in results])), 3),
                            "deadline_met": round(float(np.mean([r["elapsed"] <= r["budget"] for r in results])), 4)}
    return out


def drones(cache, indices, k=5):
    from fedorch.folds import tower_buckets
    rows = [dict(cache.meta[i], index=i) for i in indices]
    bucket = tower_buckets(rows, k)
    return [[r["index"] for r in rows if bucket[r["tower"]] == d] for d in range(k)]


def params(policy):
    return [p.detach().cpu().numpy().astype(np.float32) for p in policy.state_dict().values()]


def set_params(policy, arrays):
    policy.load_state_dict({k: torch.from_numpy(a.copy()) for k, a in zip(policy.state_dict(), arrays)})
