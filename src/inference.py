import sys
from typing import Dict, List

import numpy as np
import torch

from configs.ablation_grid import get_ablation_config, resolve_experiment_names
from configs.base_config import set_seed
from src.data import prepare_dataloaders
from src.models import build_model
from src.train import _normalize_cli_args, _parse_kv_overrides


def _evaluate_checkpoint(run_cfg: dict, checkpoint_path: str) -> None:
    device = run_cfg["device"]
    dataset_name = run_cfg["dataset_name"]
    exp_name = run_cfg["exp_name"]

    _, _, _, _, X_test, y_test, features = prepare_dataloaders(run_cfg)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    encoder = build_model(run_cfg, n_sensors=len(features))
    state = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items() if k != "n_averaged"}
    encoder.load_state_dict(state)
    encoder.eval()

    with torch.no_grad():
        preds, _, _, _ = encoder(X_test_tensor)
        preds_np = np.clip(preds.cpu().numpy(), a_min=0, a_max=run_cfg["max_rul"])

    test_rmse = float(np.sqrt(np.mean((preds_np - y_test) ** 2)))
    d = preds_np - y_test
    test_score = float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))

    print(f"[{dataset_name} | {exp_name}] Test RMSE: {test_rmse:.4f} | NASA S-Score: {test_score:.2f}")


try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _hydra_main(cfg: DictConfig) -> None:
        dataset_name = str(cfg.data.dataset_name).upper()
        seed = int(cfg.seed)
        exp_query = str(cfg.get("exp", "00_full_model"))
        exp_name = resolve_experiment_names([exp_query])[0]

        set_seed(seed)
        device_str = str(cfg.trainer.device)
        device = torch.device(device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu")

        checkpoint_path = cfg.get("checkpoint", None)
        if not checkpoint_path:
            checkpoint_path = input("Path to checkpoint (.pth): ").strip()

        run_cfg = get_ablation_config(exp_name=exp_name, dataset_name=dataset_name, seed=seed)
        run_cfg["device"] = device
        run_cfg["data_dir"] = str(cfg.data_root)
        run_cfg["max_rul"] = int(cfg.max_rul)
        run_cfg["seq_len"] = int(cfg.data.window_size)
        run_cfg["batch_size"] = int(cfg.data.batch_size)
        run_cfg["use_validation"] = False

        _evaluate_checkpoint(run_cfg, str(checkpoint_path))

    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False


def _run_without_hydra(normalized_argv: List[str]) -> None:
    ov: Dict[str, str] = _parse_kv_overrides(normalized_argv)
    dataset_name = ov.get("data", "fd001").upper()
    seed = int(ov.get("seed", "50"))
    exp_query = ov.get("exp", "00_full_model")
    exp_name = resolve_experiment_names([exp_query])[0]

    set_seed(seed)
    run_cfg = get_ablation_config(exp_name=exp_name, dataset_name=dataset_name, seed=seed)
    if "data_root" in ov:
        run_cfg["data_dir"] = ov["data_root"]
    run_cfg["use_validation"] = False

    checkpoint_path = ov.get("checkpoint", "")
    if not checkpoint_path:
        checkpoint_path = input("Path to checkpoint (.pth): ").strip()

    _evaluate_checkpoint(run_cfg, checkpoint_path)


def main() -> None:
    sys.argv = _normalize_cli_args(sys.argv)
    if HAS_HYDRA:
        _hydra_main()
    else:
        _run_without_hydra(sys.argv)


if __name__ == "__main__":
    main()
