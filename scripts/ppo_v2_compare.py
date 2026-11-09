#!/usr/bin/env python3
"""Sequential single-seed comparison; run on the VM after building a real cache."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", default="heuristic,supervised,central,local,fed")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes-per-client", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda"], default="cuda", help="GPU training only")
    parser.add_argument("--feedback", choices=["audit", "full"], default="audit")
    parser.add_argument("--budgets", default="2,6,15")
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    args, extra = parser.parse_known_args(argv)
    modes = args.modes.split(",")
    if any(m not in {"heuristic", "supervised", "central", "local", "fed"} for m in modes):
        parser.error("unknown mode")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("output must be new/empty; choose another directory to preserve pilot provenance")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    for mode in modes:
        command = [sys.executable, str(root / "scripts/ppo_v2_train.py"),
                   "--bundle", args.bundle, "--output", str(output / mode), "--mode", mode,
                   "--rounds", str(args.rounds), "--episodes-per-client", str(args.episodes_per_client),
                   "--seed", str(args.seed), "--device", args.device, "--feedback", args.feedback,
                   "--budgets", args.budgets, "--eval-split", args.eval_split]
        if args.enhance:
            command.append("--enhance")
        command.extend(extra)
        print(f"Starting {mode}: seed={args.seed}, feedback={args.feedback}, rounds={args.rounds}", flush=True)
        with (output / f"{mode}.log").open("w") as logfile:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
                logfile.flush()
            if process.wait():
                raise SystemExit(f"{mode} failed; see {output / (mode + '.log')}. Completed outputs are retained.")
    print(f"Comparison completed: {output}", flush=True)


if __name__ == "__main__":
    main()
