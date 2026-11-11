import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from fedorch.common import results_dir
from fedorch.tools_v3 import default_dtype, models_dir, run_suffix, selector_tag


def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of a small language model as tool selector (TRL SFTTrainer + LoRA).")
    parser.add_argument("--model", default=os.environ.get("TV3_SELECTOR", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--dtype", default=None, help="float32, float16 or bfloat16 (A10 and newer) for the base weights (default: float32 for 0.6B, float16 above)")
    parser.add_argument("--max-examples", type=int, default=12000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None, help="default 8 for 0.6B, 2 above")
    parser.add_argument("--grad-accum", type=int, default=None, help="default 2 for 0.6B, 8 above")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--out", default=None)
    parser.add_argument("--think", action="store_true", help="train on the steps with teacher reasons (sft_qwen_think.jsonl)")
    args = parser.parse_args()
    name = f"{selector_tag(args.model)}_sft" + ("_think" if args.think else "")
    args.out = args.out or str(models_dir() / name)
    dtype = getattr(torch, args.dtype or default_dtype(args.model))
    small = "0.6b" in args.model.lower()
    args.batch_size = args.batch_size or (8 if small else 2)
    args.grad_accum = args.grad_accum or (2 if small else 8)
    data = load_dataset("json", data_files=str(results_dir("tools_v3") / "select" / (f"sft_qwen_think{run_suffix()}.jsonl" if args.think else f"sft_qwen{run_suffix()}.jsonl")), split="train").shuffle(seed=0)
    data = data.select(range(min(args.max_examples, len(data))))
    print(f"[qwen sft] {len(data)} examples, model {args.model}", flush=True)
    cuda = torch.cuda.is_available()
    config = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
                       gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, max_length=args.max_length,
                       completion_only_loss=True, fp16=cuda and dtype != torch.bfloat16, bf16=cuda and dtype == torch.bfloat16, logging_steps=20, save_strategy="no", report_to="none",
                       gradient_checkpointing=cuda, model_init_kwargs={"dtype": dtype})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=data, peft_config=lora)
    result = trainer.train()
    trainer.save_model(args.out)
    (results_dir("tools_v3") / "select" / f"{name}_train.json").write_text(json.dumps(
        {"model": args.model, "examples": len(data), "metrics": result.metrics, "log": trainer.state.log_history}, indent=1))
    print(f"[qwen sft] saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
