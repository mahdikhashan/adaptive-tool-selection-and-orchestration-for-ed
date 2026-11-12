import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.models.utils import disable_gradient_checkpointing

from fedorch.common import results_dir
from fedorch.episode_play import play
from fedorch.route_env import Network
from fedorch.select_env import FIELDS, episode, setup
from fedorch.tools_v3 import default_dtype, excluded_tools, models_dir


def base_tag(model):
    return model.split("/")[-1].lower()


def main():
    parser = argparse.ArgumentParser(description="GRPO on full episodes: the agent plays the whole inspection of an image (tool calls and report) "
                                                 "in one conversation, and the reward is the share of correct report fields minus a time cost.")
    parser.add_argument("--model", default=os.environ.get("TV3_SELECTOR", "Qwen/Qwen3-1.7B"))
    parser.add_argument("--adapter", default=None, help="start adapter; default <model>_sft_think")
    parser.add_argument("--name", default="newtools", help="suffix of the output adapter")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--images", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--max-turn-tokens", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=5, help="tool steps plus report per episode")
    parser.add_argument("--time-cost", type=float, default=0.01, help="reward lost per second of tool time")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tag, think = base_tag(args.model), not args.no_think
    adapter = args.adapter or str(models_dir() / f"{tag}_sft_think")
    out = str(models_dir() / f"{tag}_grpo_episode_{args.name}")
    env = setup()
    rows = {r["image"]: r for r in env["val"]}
    trace = Network("weak", args.seed).trace
    rng = np.random.default_rng(args.seed)
    budgets = [4.0, 10.0, 30.0, float("inf")]
    picks = [env["val"][k] for k in rng.permutation(len(env["val"]))[: args.images]]
    data = Dataset.from_list([{"prompt": f"{r['image']}\t{budgets[rng.integers(4)]}\t{trace[rng.integers(len(trace))]}"} for r in picks])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = args.dtype or default_dtype(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, dtype))
    cuda = torch.cuda.is_available()
    if cuda:
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    stats = {"steps": [], "reward": []}

    def rollout(prompts, trainer):
        eps = []
        for p in prompts:
            image, budget, mbps = p.split("\t")
            ep = episode(env, rows[image], float(budget), float(mbps))
            ep.max_steps = args.max_steps
            eps.append(ep)
        policy = trainer.accelerator.unwrap_model(trainer.model)
        was_training = policy.training
        policy.eval()
        with disable_gradient_checkpointing(policy, trainer.args.gradient_checkpointing_kwargs):
            runs = play(policy, tokenizer, eps, think, True, args.max_turn_tokens)
        if was_training:
            policy.train()
        rewards, steps = [], []
        for r in runs:
            o = r["ep"].outcome()
            rewards.append(len(o["right"]) / len(FIELDS) - args.time_cost * o["time"])
            steps.append(len(o["called"]))
        stats["reward"].append(float(np.mean(rewards)))
        stats["steps"].append(float(np.mean(steps)))
        return {"prompt_ids": [r["prompt"] for r in runs], "completion_ids": [r["completion"] for r in runs],
                "logprobs": [r["logprobs"] for r in runs], "env_mask": [r["mask"] for r in runs],
                "episode_reward": rewards, "episode_tools": steps}

    def reward(prompts, completions, episode_reward, **kwargs):
        return [float(x) for x in episode_reward]

    accumulation = args.prompts_per_step * args.generations // args.micro_batch
    config = GRPOConfig(output_dir=out, max_steps=args.steps, per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=accumulation,
                        num_generations=args.generations, max_completion_length=16384, learning_rate=args.lr, beta=args.beta,
                        temperature=1.0, fp16=cuda and dtype == "float16", bf16=cuda and dtype == "bfloat16", logging_steps=1, save_strategy="steps", save_steps=25, save_total_limit=1,
                        report_to="none", gradient_checkpointing=cuda, seed=args.seed)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=config, train_dataset=data, processing_class=tokenizer, rollout_func=rollout)
    print(f"[grpo episode] {args.model}, start adapter {adapter}, tools {len(env['tools'])} (excluded: {sorted(excluded_tools()) or 'none'}), "
          f"{len(data)} images, {args.steps} steps, {args.prompts_per_step} images x {args.generations} episodes per step, think {think}", flush=True)
    trainer.train()
    trainer.save_model(out)
    (results_dir("tools_v3") / "select" / f"{tag}_grpo_episode_{args.name}_train.json").write_text(
        json.dumps({"log": trainer.state.log_history, "rollout": stats, "args": vars(args)}, indent=1))
    print(f"[grpo episode] saved adapter to {out}", flush=True)


if __name__ == "__main__":
    main()
