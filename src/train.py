import os
import sys
from typing import Any, Dict, List

import torch

from configs.ablation_grid import get_ablation_config, resolve_experiment_names
from configs.base_config import get_dataset_config
from src.trainer import train_and_evaluate_single_run


def _normalize_cli_args(argv: List[str]) -> List[str]:
    """
    Позволяет передавать как стандартные Hydra-аргументы (data=fd001 seed=50),
    так и привычные CLI-флаги через '--' (--dataset FD001 --seed 50 --exp 0 --epochs 24 и т.д.).
    """
    flag_map = {
        "--dataset": "data",
        "--data": "data",
        "--seed": "seed",
        "--exp": "exp",
        "--data_dir": "data_root",
        "--data_root": "data_root",
        "--out_dir": "out_dir",
        "--save_dir": "out_dir",
        "--max_rul": "max_rul",
        "--epochs": "trainer.epochs",
        "--ema_decay": "trainer.ema_decay",
        "--kl_weight": "trainer.kl_weight",
        "--mse_weight": "trainer.mse_weight",
        "--lambda_order": "trainer.lambda_order",
        "--use_proxy_1k": "trainer.use_proxy_1k",
        "--norm_mode": "trainer.norm_mode",
        "--use_validation": "trainer.use_validation",
        "--val_split": "trainer.val_split",
        "--save_epoch": "trainer.save_epoch",
        "--device": "trainer.device",
        "--num_proxies": "model.num_proxies",
        "--temp": "model.temp",
        "--emb_dim": "model.emb_dim",
        "--patch_size": "model.patch_size",
        "--sensor_pool": "model.sensor_pool",
        "--head_type": "model.head_type",
        "--use_residual": "model.use_residual",
        "--lr": "data.lr",
        "--weight_decay": "data.weight_decay",
        "--batch_size": "data.batch_size",
        "--seq_len": "data.window_size",
        "--window_size": "data.window_size",
        "--d_model": "data.d_model",
        "--nhead": "data.nhead",
        "--num_scales": "data.num_scales",
        "--ffn_dim": "data.ffn_dim",
        "--dropout": "data.dropout",
    }
    bool_flags = {
        "--use_validation": ("trainer.use_validation", "true"),
        "--no_plots": ("trainer.save_plots", "false"),
        "--save_plots": ("trainer.save_plots", "true"),
        "--save_weights": ("trainer.save_weights", "true"),
        "--no_weights": ("trainer.save_weights", "false"),
    }

    normalized: List[str] = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help", "--cfg", "--info", "--multirun", "-m"):
            normalized.append(arg)
            i += 1
            continue

        if arg.startswith("--") and "=" in arg:
            k, v = arg.split("=", 1)
            mapped_k = flag_map.get(k, k.lstrip("-"))
            if mapped_k == "data":
                v = v.lower()
            normalized.append(f"{mapped_k}={v}")
            i += 1
            continue

        if arg in bool_flags:
            mapped_k, default_bool = bool_flags[arg]
            if i + 1 < len(argv) and argv[i + 1].lower() in ("true", "false", "1", "0"):
                val = "true" if argv[i + 1].lower() in ("true", "1") else "false"
                normalized.append(f"{mapped_k}={val}")
                i += 2
            else:
                normalized.append(f"{mapped_k}={default_bool}")
                i += 1
            continue

        if arg in flag_map and i + 1 < len(argv):
            mapped_k = flag_map[arg]
            val = argv[i + 1]
            if mapped_k == "data":
                val = val.lower()
            normalized.append(f"{mapped_k}={val}")
            i += 2
            continue

        normalized.append(arg)
        i += 1

    return normalized


def _parse_kv_overrides(normalized_argv: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in normalized_argv[1:]:
        if "=" in item and not item.startswith("-"):
            k, v = item.split("=", 1)
            overrides[k.strip()] = v.strip()
    return overrides


def _str_to_bool(val: str) -> bool:
    return val.lower() in ("true", "1", "yes")


def _run_without_hydra(normalized_argv: List[str]) -> None:
    if any(a in ("-h", "--help") for a in normalized_argv[1:]):
        print(
            "Usage: python -m src.train [--dataset FD001] [--seed 50] [--exp 00_full_model] [--epochs N] "
            "[data=fd001] [seed=50] [exp=00_full_model] ..."
        )
        return

    ov = _parse_kv_overrides(normalized_argv)
    dataset_name = ov.get("data", "fd001").upper()
    seed = int(ov.get("seed", "50"))
    exp_query = ov.get("exp", "00_full_model")
    exp_name = resolve_experiment_names([exp_query])[0]

    run_cfg = get_ablation_config(exp_name=exp_name, dataset_name=dataset_name, seed=seed)
    is_kaggle = os.path.exists("/kaggle")
    default_out = "/kaggle/working/results" if is_kaggle else "results"

    if "data_root" in ov:
        run_cfg["data_dir"] = ov["data_root"]
    out_dir = ov.get("out_dir", default_out)
    run_cfg["save_dir"] = out_dir

    if "max_rul" in ov:
        run_cfg["max_rul"] = int(ov["max_rul"])
    if "trainer.device" in ov:
        dev_str = ov["trainer.device"]
        run_cfg["device"] = torch.device(dev_str if torch.cuda.is_available() and dev_str != "cpu" else "cpu")

    int_map = {
        "data.window_size": "seq_len",
        "data.batch_size": "batch_size",
        "trainer.epochs": "epochs",
        "data.epochs": "epochs",
        "data.d_model": "d_model",
        "data.nhead": "nhead",
        "data.num_scales": "num_scales",
        "data.ffn_dim": "ffn_dim",
        "model.emb_dim": "emb_dim",
        "model.num_proxies": "num_proxies",
        "model.patch_size": "patch_size",
    }
    for k, target in int_map.items():
        if k in ov and ov[k].lower() != "null":
            run_cfg[target] = int(ov[k])

    float_map = {
        "data.lr": "lr",
        "data.weight_decay": "weight_decay",
        "data.dropout": "dropout",
        "model.temp": "temp",
        "trainer.ema_decay": "ema_decay",
        "trainer.kl_weight": "kl_weight",
        "trainer.mse_weight": "mse_weight",
        "trainer.lambda_order": "lambda_order",
        "trainer.val_split": "val_split",
    }
    for k, target in float_map.items():
        if k in ov and ov[k].lower() != "null":
            run_cfg[target] = float(ov[k])

    bool_map = {
        "model.use_residual": "use_residual",
        "trainer.use_proxy_1k": "use_proxy_1k",
        "trainer.use_validation": "use_validation",
    }
    for k, target in bool_map.items():
        if k in ov:
            run_cfg[target] = _str_to_bool(ov[k])

    str_map = {
        "model.sensor_pool": "sensor_pool",
        "model.head_type": "head_type",
        "trainer.norm_mode": "norm_mode",
    }
    for k, target in str_map.items():
        if k in ov:
            run_cfg[target] = ov[k]

    if "trainer.save_epoch" in ov and ov["trainer.save_epoch"].lower() != "null":
        run_cfg["save_epoch"] = int(ov["trainer.save_epoch"])

    save_weights = _str_to_bool(ov.get("trainer.save_weights", "true"))
    save_plots = _str_to_bool(ov.get("trainer.save_plots", "true"))
    enable_tb = _str_to_bool(ov.get("trainer.enable_tb", "true"))

    train_and_evaluate_single_run(
        cfg=run_cfg,
        out_dir=out_dir,
        save_weights=save_weights,
        save_plots=save_plots,
        enable_tb=enable_tb,
    )


try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _hydra_main(cfg: DictConfig) -> None:
        dataset_name = str(cfg.data.dataset_name).upper()
        seed = int(cfg.seed)
        exp_query = str(cfg.get("exp", "00_full_model"))
        exp_name = resolve_experiment_names([exp_query])[0]

        run_cfg = get_ablation_config(exp_name=exp_name, dataset_name=dataset_name, seed=seed)
        default_base = get_dataset_config(dataset_name)

        device_str = str(cfg.trainer.device)
        run_cfg["device"] = torch.device(
            device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu"
        )
        run_cfg["data_dir"] = str(cfg.data_root)
        run_cfg["save_dir"] = str(cfg.out_dir)
        run_cfg["max_rul"] = int(cfg.max_rul)
        run_cfg["seq_len"] = int(cfg.data.window_size)
        run_cfg["batch_size"] = int(cfg.data.batch_size)
        run_cfg["epochs"] = (
            int(cfg.trainer.epochs) if cfg.trainer.epochs is not None else int(cfg.data.epochs)
        )
        run_cfg["lr"] = float(cfg.data.lr)
        run_cfg["weight_decay"] = float(cfg.data.weight_decay)
        run_cfg["d_model"] = int(cfg.data.d_model)
        run_cfg["nhead"] = int(cfg.data.nhead)
        run_cfg["num_scales"] = int(cfg.data.num_scales)
        run_cfg["ffn_dim"] = int(cfg.data.ffn_dim)
        run_cfg["dropout"] = float(cfg.data.dropout)

        run_cfg["emb_dim"] = int(cfg.model.emb_dim)
        run_cfg["num_proxies"] = int(cfg.model.num_proxies)
        run_cfg["temp"] = float(cfg.model.temp)
        run_cfg["patch_size"] = int(cfg.model.patch_size)
        run_cfg["sensor_pool"] = str(cfg.model.sensor_pool)
        run_cfg["ema_decay"] = float(cfg.trainer.ema_decay)
        run_cfg["use_validation"] = bool(cfg.trainer.use_validation)
        run_cfg["val_split"] = float(cfg.trainer.val_split)
        run_cfg["save_epoch"] = cfg.trainer.save_epoch

        if str(cfg.model.head_type) != default_base["head_type"]:
            run_cfg["head_type"] = str(cfg.model.head_type)
        if bool(cfg.model.use_residual) != default_base["use_residual"]:
            run_cfg["use_residual"] = bool(cfg.model.use_residual)
        if float(cfg.trainer.kl_weight) != default_base["kl_weight"]:
            run_cfg["kl_weight"] = float(cfg.trainer.kl_weight)
        if float(cfg.trainer.mse_weight) != default_base["mse_weight"]:
            run_cfg["mse_weight"] = float(cfg.trainer.mse_weight)
        if float(cfg.trainer.lambda_order) != default_base["lambda_order"]:
            run_cfg["lambda_order"] = float(cfg.trainer.lambda_order)
        if bool(cfg.trainer.use_proxy_1k) != default_base["use_proxy_1k"]:
            run_cfg["use_proxy_1k"] = bool(cfg.trainer.use_proxy_1k)
        if str(cfg.trainer.norm_mode) != default_base["norm_mode"]:
            run_cfg["norm_mode"] = str(cfg.trainer.norm_mode)

        train_and_evaluate_single_run(
            cfg=run_cfg,
            out_dir=str(cfg.out_dir),
            save_weights=bool(cfg.trainer.get("save_weights", True)),
            save_plots=bool(cfg.trainer.get("save_plots", True)),
            enable_tb=bool(cfg.trainer.get("enable_tb", True)),
        )

    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False


def main() -> None:
    sys.argv = _normalize_cli_args(sys.argv)
    if HAS_HYDRA:
        _hydra_main()
    else:
        _run_without_hydra(sys.argv)


if __name__ == "__main__":
    main()
