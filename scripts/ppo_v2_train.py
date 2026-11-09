#!/usr/bin/env python3
"""Run the frozen-cache PPO v2 controller pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fedorch.ppo_v2_train import main
if __name__ == '__main__':
    main()
