import random
from typing import Any, Dict, List

import numpy as np
import torch

DATASETS: List[str] = ["FD001", "FD002", "FD003", "FD004"]

DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 50]

COLS: List[str] = ["unit", "time", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
DROP_SENSORS: List[str] = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of {DATASETS}.")

    base: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "data_dir": "/kaggle/input/datasets/kylibiaka/cmapss",
        "save_dir": "/kaggle/working/results",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "max_rul": 125,
        "ema_decay": 0.99,
        "seed": 50,
        "use_validation": False,
        "val_split": 0.15,
        "num_proxies": 16,
        "temp": 0.02,
        "sensor_pool": "mean",
        "head_type": "proxy",          # 'proxy' | 'linear'
        "use_residual": True,          # Decoupled Residual Proxy Head (delta_rul)
        "use_proxy_1k": True,          # Proxy-Centric 1/K Aggregation vs Standard 1/B Batch-Mean
        "norm_mode": "regime",         # 'regime' (6 clusters on FD002/FD004, 1 on FD001/FD003) | 'global' (1 cluster)
        "kl_weight": 1.0,
        "mse_weight": 0.05,
        "lambda_order": 0.5,
    }

    fd_params: Dict[str, Dict[str, Any]] = {
        "FD001": {
            "seq_len": 32,
            "batch_size": 32,
            "epochs": 24,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "emb_dim": 64,
            "d_model": 64,
            "nhead": 1,
            "num_scales": 2,
            "ffn_dim": 128,
            "patch_size": 4,
            "dropout": 0.1,
        },
        "FD002": {
            "seq_len": 64,
            "batch_size": 64,
            "epochs": 16,
            "lr": 2e-4,
            "weight_decay": 1e-2,
            "emb_dim": 64,
            "d_model": 128,
            "nhead": 4,
            "num_scales": 3,
            "ffn_dim": 256,
            "patch_size": 4,
            "dropout": 0.2,
        },
        "FD003": {
            "seq_len": 48,
            "batch_size": 32,
            "epochs": 20,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "emb_dim": 64,
            "d_model": 64,
            "nhead": 1,
            "num_scales": 2,
            "ffn_dim": 128,
            "patch_size": 4,
            "dropout": 0.1,
        },
        "FD004": {
            "seq_len": 64,
            "batch_size": 64,
            "epochs": 16,
            "lr": 2e-4,
            "weight_decay": 1e-2,
            "emb_dim": 64,
            "d_model": 128,
            "nhead": 4,
            "num_scales": 3,
            "ffn_dim": 256,
            "patch_size": 4,
            "dropout": 0.2,
        },
    }

    base.update(fd_params[dataset_name])
    return base
