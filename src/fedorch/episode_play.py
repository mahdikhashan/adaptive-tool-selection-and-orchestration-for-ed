import torch

from fedorch.select_env import messages, parse, summary


def first_prompt(tokenizer, ep, actions, think):
    return tokenizer.apply_chat_template(messages(ep, actions), tokenize=False, add_generation_prompt=True, enable_thinking=think)


def next_turn(ep, actions, think, tool):
    candidates = "\n\n".join(f"[{a}]\n{ep.doc(a)}" for a in actions)
    content = f"Result of {tool}: {summary(tool, ep.r['out'][tool])}\n\nSub-step: {ep.substep()}\n\nCandidates ({len(actions)}):\n{candidates}"
    return f"\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n" + ("" if think else "<think>\n\n</think>\n\n")


def play(model, tokenizer, eps, think, sample, max_new_tokens, max_tokens=12000):
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    runs = []
    for ep in eps:
        actions = ep.candidates()
        runs.append({"ep": ep, "actions": actions, "prompt": tokenizer(first_prompt(tokenizer, ep, actions, think), add_special_tokens=False)["input_ids"],
                     "completion": [], "mask": [], "logprobs": []})
    while True:
        active = [r for r in runs if not r["ep"].done]
        if not active:
            break
        seqs = [r["prompt"] + r["completion"] for r in active]
        width = max(map(len, seqs))
        ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=model.device)
        attention = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=model.device)
        options = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=eos,
                                 return_dict_in_generate=True, output_scores=sample, **options)
        new = out.sequences[:, width:]
        scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True) if sample else torch.zeros_like(new, dtype=torch.float)
        for r, row, lp in zip(active, new.tolist(), scores.tolist()):
            ended = eos in row
            n = row.index(eos) + 1 if ended else len(row)
            r["completion"] += row[:n]
            r["mask"] += [1] * n
            r["logprobs"] += [float(x) for x in lp[:n]]
            ep = r["ep"]
            choice = parse(tokenizer.decode(row[:n], skip_special_tokens=True), r["actions"]) if ended else None
            ep.step(choice)
            if ep.done:
                continue
            r["actions"] = ep.candidates()
            if len(r["prompt"]) + len(r["completion"]) > max_tokens:
                r["actions"] = ["report"]
            env = tokenizer(next_turn(ep, r["actions"], think, choice), add_special_tokens=False)["input_ids"]
            r["completion"] += env
            r["mask"] += [0] * len(env)
            r["logprobs"] += [0.0] * len(env)
    return runs
