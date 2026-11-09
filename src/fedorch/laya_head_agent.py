"""Base laya as the agent, option 2: PPO trains laya's head transformer, type embedding and scorers
on cached frozen backbone outputs (runs/agent_cache_h). Environment and reward as in laya_agent."""
import json
import math

import numpy as np
import torch
from torch import nn

from fedorch.laya_agent import ACTIONS, MISS, FALSE_ALARM, REPORT, STATE, Episode, advantages, macro_f1, report_cost

SIZES = (8, 11, 2, 6)
STATES = [(v, r, c) for v in range(6) for r in (0, 1) for c in (0, 1)]


class HeadCache:
    def __init__(self, source, root):
        self.meta = [json.loads(l) for l in open(source / "meta.jsonl")]
        names = [m["file"].replace(".npy", "") for m in self.meta]
        self.h = [np.load(root / f"{n}_h.npy", mmap_mode="r") for n in names]
        self.aux = [dict(np.load(root / f"{n}_aux.npz")) for n in names]
        self.vectors = [None] * len(self.meta)
        self.base = torch.load(root / "base_head.pt", map_location="cpu")

    def batch(self, keys, device):
        """keys: list of (image index, view, resnet shown, qwen shown) -> padded tensors for 4 rows each."""
        rows = [(self.h[i][STATES.index((v, r, c))], {k: a[STATES.index((v, r, c))] for k, a in self.aux[i].items()})
                for i, v, r, c in keys]
        length = max(h.shape[1] for h, _ in rows)
        markers = max(a["pos"].shape[1] for _, a in rows)
        n = len(rows)
        h = torch.zeros((n, 4, length, 576), dtype=torch.float16)
        mask = torch.zeros((n, 4, length), dtype=torch.bool)
        pos = torch.zeros((n, 4, markers), dtype=torch.long)
        mmask = torch.zeros((n, 4, markers), dtype=torch.bool)
        qtype = torch.zeros((n, 4), dtype=torch.long)
        for j, (hh, a) in enumerate(rows):
            h[j, :, :hh.shape[1]] = torch.from_numpy(np.asarray(hh))
            mask[j, :, :hh.shape[1]] = torch.from_numpy(a["mask"])
            pos[j, :, :a["pos"].shape[1]] = torch.from_numpy(a["pos"].astype(np.int64))
            mmask[j, :, :a["mmask"].shape[1]] = torch.from_numpy(a["mmask"])
            qtype[j] = torch.from_numpy(a["qtype"].astype(np.int64))
        return [t.to(device) for t in (h, mask, pos, mmask, qtype)]


def scorer(state):
    net = nn.Sequential(nn.LayerNorm(576), nn.Linear(576, 576), nn.GELU(), nn.Linear(576, 1))
    net.load_state_dict(state)
    return net


class HeadPolicy(nn.Module):
    def __init__(self, base):
        super().__init__()
        layer = nn.TransformerEncoderLayer(576, 9, 4 * 576, 0., batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head.load_state_dict(base["head"])
        self.type_emb = nn.Embedding(3, 576)
        self.type_emb.load_state_dict(base["type_emb"])
        self.action_scorer, self.report_scorer = scorer(base["scorer"]), scorer(base["scorer"])
        self.state = nn.Sequential(nn.Linear(STATE, 32), nn.Tanh(), nn.Linear(32, ACTIONS))
        nn.init.zeros_(self.state[2].weight)
        nn.init.zeros_(self.state[2].bias)

    def markers(self, h, mask, pos, mmask, qtype):
        n, rows, length, d = h.shape
        x = h.reshape(n * rows, length, d).float() + self.type_emb(qtype.reshape(-1))[:, None, :]
        x = self.head(x, src_key_padding_mask=~mask.reshape(n * rows, length))
        idx = pos.reshape(n * rows, -1).clamp(min=0)[:, :, None].expand(-1, -1, d)
        return torch.gather(x, 1, idx).reshape(n, rows, -1, d)

    def logits(self, batch, state):
        h, mask, pos, mmask, qtype = batch
        m = self.markers(h, mask, pos, mmask, qtype)
        action = self.action_scorer(m[:, 0, :ACTIONS]).squeeze(-1) + self.state(state)
        report = [self.report_scorer(m[:, q, :k]).squeeze(-1) for q, k in ((1, SIZES[1]), (2, SIZES[2]), (3, SIZES[3]))]
        return action, report, m[:, 0, :ACTIONS].mean(1).detach()


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(576 + STATE, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, pooled, state):
        return self.net(torch.cat([pooled, state], -1)).squeeze(-1)


def rollout(policy, critic, cache, indices, budgets, rng, sample=True, tools=True, lambda_time=1., device="cuda"):
    episodes, results = [], []
    for index in indices:
        budget = budgets[rng.integers(len(budgets))] if sample else budgets
        ep = Episode(cache, index, budget, tools=tools)
        steps = []
        while True:
            key = (index, ep.view, ep.resnet, ep.qwen)
            x, mask = ep.features().to(device)[None], ep.mask().to(device)
            with torch.no_grad():
                action_logits, report_logits, pooled = policy.logits(cache.batch([key], device), x)
                dist = torch.distributions.Categorical(logits=action_logits[0].masked_fill(~mask, -1e9))
                action = int(dist.sample()) if sample else int(dist.probs.argmax())
                logp = float(dist.log_prob(torch.tensor(action, device=device)))
                value = float(critic(pooled, x)[0])
                answers = []
                if action == REPORT:
                    for logits in report_logits:
                        d = torch.distributions.Categorical(logits=logits[0])
                        choice = int(d.sample()) if sample else int(logits[0].argmax())
                        answers.append(choice)
                        logp += float(d.log_prob(torch.tensor(choice, device=device)))
            duration = ep.step(action)
            reward = -lambda_time * duration / ep.budget
            if action == REPORT:
                cost = report_cost(ep.m, *answers)
                reward -= cost
                results.append({"budget": ep.budget, "elapsed": ep.elapsed, "steps": ep.steps, "component": answers[0],
                                "defect": answers[1], "condition": answers[2], "cost": cost,
                                "truth": [ep.m["component"], ep.m["defect"], ep.m["condition"]]})
            steps.append(dict(key=key, x=x[0].cpu(), mask=mask.cpu(), action=action, answers=answers, logp=logp,
                              value=value, reward=reward, duration=duration, done=action == REPORT))
            if action == REPORT:
                break
        episodes.append(steps)
    return episodes, results


def update(policy, critic, episodes, cache, epochs=3, lr=5e-5, clip=.2, entropy=.01, prox=0., anchor=None, device="cuda", minibatch=64):
    steps = [s for ep in episodes for s in ep]
    adv = torch.tensor([a for ep in episodes for a in advantages(ep)])
    returns = adv + torch.tensor([s["value"] for s in steps])
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    opt_p = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(epochs):
        for ids in torch.randperm(len(steps)).split(minibatch):
            chunk = [steps[i] for i in ids.tolist()]
            x = torch.stack([s["x"] for s in chunk]).to(device)
            mask = torch.stack([s["mask"] for s in chunk]).to(device)
            actions = torch.tensor([s["action"] for s in chunk], device=device)
            action_logits, report_logits, pooled = policy.logits(cache.batch([s["key"] for s in chunk], device), x)
            dist = torch.distributions.Categorical(logits=action_logits.masked_fill(~mask, -1e9))
            logp, ent = dist.log_prob(actions), dist.entropy()
            is_report = (actions == REPORT).float()
            for q, logits in enumerate(report_logits):
                d = torch.distributions.Categorical(logits=logits)
                chosen = torch.tensor([s["answers"][q] if s["answers"] else 0 for s in chunk], device=device)
                logp, ent = logp + is_report * d.log_prob(chosen), ent + is_report * d.entropy()
            ratio = (logp - torch.tensor([s["logp"] for s in chunk], device=device)).exp()
            a = adv[ids].to(device)
            loss = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean() - entropy * ent.mean()
            if prox:
                loss = loss + prox / 2 * sum(((p - q) ** 2).sum() for p, q in zip(policy.parameters(), anchor))
            opt_p.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), .5); opt_p.step()
            value_loss = ((critic(pooled, x) - returns[ids].to(device)) ** 2).mean()
            opt_c.zero_grad(); value_loss.backward(); opt_c.step()
    return {"policy_loss": float(loss.detach()), "value_loss": float(value_loss.detach()), "transitions": len(steps)}


def evaluate(policy, cache, indices, budgets, tools=True, device="cuda"):
    out, critic = {}, Critic().to(device)
    for budget in budgets:
        _, results = rollout(policy, critic, cache, indices, budget, np.random.default_rng(0), sample=False, tools=tools, device=device)
        truth = np.array([r["truth"] for r in results])
        pred = np.array([[r["component"], r["defect"], r["condition"]] for r in results])
        defects = truth[:, 1] == 1
        miss, false_alarm = np.sum(defects & (pred[:, 1] == 0)), np.sum(~defects & (pred[:, 1] == 1))
        out[str(budget)] = {"images": len(results),
                            "report_cost_per_1000": round(1000 * float(np.mean([r["cost"] for r in results])), 1),
                            "defect_risk_per_1000": round(1000 * (MISS * miss + FALSE_ALARM * false_alarm) / len(results), 1),
                            "defect_recall": round(float(np.mean(pred[defects, 1] == 1)), 4),
                            "false_alarm": round(float(np.mean(pred[~defects, 1] == 1)), 4),
                            "component_macro_f1": round(macro_f1(truth[:, 0], pred[:, 0], 11), 4),
                            "condition_macro_f1": round(macro_f1(truth[:, 2], pred[:, 2], 6), 4),
                            "mean_steps": round(float(np.mean([r["steps"] for r in results])), 2),
                            "mean_time_s": round(float(np.mean([r["elapsed"] for r in results])), 3),
                            "used_a_tool": round(float(np.mean([r["steps"] > 1 for r in results])), 3)}
    return out
