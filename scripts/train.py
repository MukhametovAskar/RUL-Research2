#!/usr/bin/env python3
"""
Entrypoint for single-run training using Hydra or CLI flags.

Examples:
    python scripts/train.py experiment=03_wo_ldl_kl data=fd001 seed=42
    python scripts/train.py experiment=00_full_model data=fd002 seed=50 trainer.use_validation=true
    python scripts/train.py --dataset FD001 --seed 50 --exp 00_full_model
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.train import main

if __name__ == "__main__":
    main()
