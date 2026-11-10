import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from fedorch.common import results_dir
from fedorch.tools_v3 import models_dir


def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of a small language model as tool selector (TRL SFTTrainer + LoRA).")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-examples", type=int, default=12000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--out", default=str(models_dir() / "qwen_sft"))
    args = parser.parse_args()
    data = load_dataset("json", data_files=str(results_dir("tools_v3") / "select" / "sft_qwen.jsonl"), split="train").shuffle(seed=0)
    data = data.select(range(min(args.max_examples, len(data))))
    print(f"[qwen sft] {len(data)} examples, model {args.model}", flush=True)
    cuda = torch.cuda.is_available()
    config = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
                       gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, max_length=args.max_length,
                       completion_only_loss=True, fp16=cuda, logging_steps=20, save_strategy="no", report_to="none",
                       gradient_checkpointing=cuda, model_init_kwargs={"dtype": torch.float32})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=data, peft_config=lora)
    result = trainer.train()
    trainer.save_model(args.out)
    (results_dir("tools_v3") / "select" / "qwen_sft_train.json").write_text(json.dumps(
        {"model": args.model, "examples": len(data), "metrics": result.metrics, "log": trainer.state.log_history}, indent=1))
    print(f"[qwen sft] saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
