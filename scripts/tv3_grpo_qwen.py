import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from fedorch.common import results_dir
from fedorch.select_env import FIELDS, correct_fields, parse
from fedorch.tools_v3 import default_dtype, models_dir, selector_tag


def reward(prompts, completions, candidates, truth, gold, **kwargs):
    out = []
    for c, a, t, g in zip(completions, candidates, truth, gold):
        choice = parse(c, json.loads(a))
        if isinstance(choice, dict):
            out.append(len(correct_fields(choice, json.loads(t))) / len(FIELDS) if g == "report" else 0.)
        else:
            out.append(float(choice is not None and choice == g))
    return out


def main():
    parser = argparse.ArgumentParser(description="GRPO on the SFT inspection agent (TRL GRPOTrainer, LoRA continues from the SFT adapter).")
    parser.add_argument("--model", default=os.environ.get("TV3_SELECTOR", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-prompts", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None, help="completions per step; default 16 for 0.6B, 8 above")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--out", default=None)
    parser.add_argument("--think", action="store_true", help="start from the thinking SFT adapter; prompts with thinking on; longer completions")
    parser.add_argument("--max-completion", type=int, default=None, help="default 96, or 384 with --think")
    args = parser.parse_args()
    tag = selector_tag(args.model)
    args.batch_size = args.batch_size or (16 if "0.6b" in args.model.lower() else 8)
    suffix = "_think" if args.think else ""
    args.adapter = args.adapter or str(models_dir() / f"{tag}_sft{suffix}")
    args.out = args.out or str(models_dir() / f"{tag}_grpo{suffix}")
    args.max_completion = args.max_completion or (384 if args.think else 96)
    data = load_dataset("json", data_files=str(results_dir("tools_v3") / "select" / "grpo_qwen.jsonl"), split="train").shuffle(seed=0)
    data = data.select(range(min(args.max_prompts, len(data))))
    if args.think:
        data = data.map(lambda r: {"prompt": r["prompt"].replace("<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n")})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype or default_dtype(args.model))
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    cuda = torch.cuda.is_available()
    if cuda:
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    config = GRPOConfig(output_dir=args.out, max_steps=args.steps, per_device_train_batch_size=args.batch_size,
                        num_generations=args.generations, max_completion_length=args.max_completion, learning_rate=args.lr, beta=args.beta,
                        temperature=1.0, fp16=cuda and dtype != torch.bfloat16, bf16=cuda and dtype == torch.bfloat16, logging_steps=5, save_strategy="no", report_to="none",
                        gradient_checkpointing=cuda)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=config, train_dataset=data, processing_class=tokenizer)
    print(f"[qwen grpo] {len(data)} prompts, {args.steps} steps, {args.generations} generations each ", flush=True)
    trainer.train()
    trainer.save_model(args.out)
    (results_dir("tools_v3") / "select" / f"{tag}_grpo{suffix}_train.json").write_text(json.dumps({"log": trainer.state.log_history}, indent=1))
    print(f"[qwen grpo] saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
