"""Backbone cache for training base laya's head: for every image in runs/agent_cache and every
(view, ResNet shown, Qwen shown) state, store the frozen 4-bit base-laya backbone output for the four
questions (action, component, defect, condition). The head transformer and scorers then train on it.
Reuses the tool outputs in runs/agent_cache/meta.jsonl; no tool runs again.
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fedorch import quant
from fedorch.common import runs_dir
from fedorch.items import INSPLAD

spec = importlib.util.spec_from_file_location("agent_cache", Path(__file__).with_name("agent_cache.py"))
agent_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_cache)


def load_base():
    from laya.vlm import load_vlm
    from transformers import AutoModel
    folder = quant.STORE / "laya_base_nf4_fp16"
    agent = load_vlm(str(folder), device="cuda", dtype="fp32")
    agent.model.encoder = quant.load(AutoModel, folder / "backbone_quantized", attn_implementation="sdpa")
    agent._torch_dtype = lambda: torch.float16
    captured = []
    original = agent.model._readout

    def capture(h, attention_mask, marker_pos, marker_mask, qtype):
        captured.append((h.detach().half().cpu(), attention_mask.bool().cpu(), marker_pos.cpu(), marker_mask.cpu(), qtype.cpu()))
        return original(h, attention_mask, marker_pos, marker_mask, qtype)

    agent.model._readout = capture
    return agent, captured


def main():
    source = runs_dir("agent_cache")
    out = runs_dir("agent_cache_h")
    meta = [json.loads(l) for l in (source / "meta.jsonl").open()]
    agent, captured = load_base()
    m = agent.model
    torch.save({"type_emb": m.type_emb.state_dict(), "head": m.head.state_dict(), "scorer": m.scorer.state_dict()}, out / "base_head.pt")
    states = [(v, r, c) for v in range(len(agent_cache.VIEWS)) for r in (0, 1) for c in (0, 1)]
    start, todo = time.time(), [x for x in meta if not (out / x["file"].replace(".npy", "_h.npy")).exists()]
    print(f"{len(todo)} images, {len(states)} states each", flush=True)
    for n, row in enumerate(todo, 1):
        image = Image.open(INSPLAD / row["image"]).convert("RGB")
        views, _ = agent_cache.views(image)
        top = [", ".join(f"{k} {p:.2f}" for k, p in sorted(probs.items(), key=lambda kv: -kv[1])[:2]) for probs in row["resnet"]]
        parts = []
        for v, r, c in states:
            evidence = {}
            if r:
                evidence["onboard defect classifier on this view (prediction, not truth)"] = top[v]
            if c:
                evidence["cloud vision-language model on the full image (prediction, not truth)"] = row["qwen_text"]
            state = {"image": views[v], **({"evidence": evidence} if evidence else {})}
            captured.clear()
            with torch.inference_mode():
                agent.predict(state, agent_cache.QUESTIONS, prefix_cache=False)
            if len(captured) != 1 or captured[0][0].shape[0] != 4:
                raise RuntimeError("expected one readout call with four rows")
            parts.append(captured[0])
        length = max(p[0].shape[1] for p in parts)
        markers = max(p[2].shape[1] for p in parts)
        h = np.zeros((len(states), 4, length, 576), np.float16)
        mask = np.zeros((len(states), 4, length), bool)
        pos = np.zeros((len(states), 4, markers), np.int16)
        mmask = np.zeros((len(states), 4, markers), bool)
        qtype = np.zeros((len(states), 4), np.int8)
        for s, (hh, am, mp, mm, qt) in enumerate(parts):
            h[s, :, :hh.shape[1]], mask[s, :, :am.shape[1]] = hh.numpy(), am.numpy()
            pos[s, :, :mp.shape[1]], mmask[s, :, :mm.shape[1]], qtype[s] = mp.numpy(), mm.numpy(), qt.numpy()
        base = row["file"].replace(".npy", "")
        np.savez(out / f"{base}_aux.npz", mask=mask, pos=pos, mmask=mmask, qtype=qtype)
        np.save(out / f"{base}_h.npy", h)
        if n % 25 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} images, {(time.time() - start) / n:.2f} s each", flush=True)
    print(f"done, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
