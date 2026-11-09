#!/usr/bin/env python3
"""Local browser demo; importing this entry point requires only the standard library."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedorch.agentic_demo import main

if __name__ == "__main__":
    main()
