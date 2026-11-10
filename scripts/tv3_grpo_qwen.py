import argparse
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from fedorch.common import results_dir
from fedorch.select_env import parse
from fedorch.tools_v3 import models_dir

def exact_reward(prompts, completions, values, gold, **kwargs):
    return [float(parse(c, list(json.loads(v))) == g) for c, v, g in zip(completions, values, gold)]


def risk_reward(prompts, completions, values, **kwargs):
    out = []
    for c, v in zip(completions, values):
        table = json.loads(v)
        choice = parse(c, list(table))
        out.append(-table[choice] if choice is not None else -3.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="GRPO on the SFT tool selector (TRL GRPOTrainer, LoRA continues from the SFT adapter).")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=str(models_dir() / "qwen_sft"))
    parser.add_argument("--max-prompts", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--reward", choices=["exact", "risk"], default="exact", help="exact match with the gold tool (as in the paper) or minus the risk")
    parser.add_argument("--out", default=str(models_dir() / "qwen_grpo"))
    args = parser.parse_args()
    data = load_dataset("json", data_files=str(results_dir("tools_v3") / "select" / "grpo_qwen.jsonl"), split="train").shuffle(seed=0)
    data = data.select(range(min(args.max_prompts, len(data))))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    cuda = torch.cuda.is_available()
    if cuda:
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    config = GRPOConfig(output_dir=args.out, max_steps=args.steps, per_device_train_batch_size=args.batch_size,
                        num_generations=args.generations, max_completion_length=32, learning_rate=args.lr, beta=args.beta,
                        temperature=1.0, fp16=cuda, logging_steps=5, save_strategy="no", report_to="none",
                        gradient_checkpointing=cuda)
    trainer = GRPOTrainer(model=model, reward_funcs=exact_reward if args.reward == "exact" else risk_reward, args=config, train_dataset=data, processing_class=tokenizer)
    print(f"[qwen grpo] {len(data)} prompts, {args.steps} steps, {args.generations} generations each, reward {args.reward}", flush=True)
    trainer.train()
    trainer.save_model(args.out)
    (results_dir("tools_v3") / "select" / "qwen_grpo_train.json").write_text(json.dumps({"log": trainer.state.log_history}, indent=1))
    print(f"[qwen grpo] saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
