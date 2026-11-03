import csv
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from configs.base_config import set_seed
from .data import CMAPSSDataset, prepare_dataloaders
from .losses import build_criterion
from .models import build_model, build_optimizer
from .visualization import (
    compute_smart_pccs,
    compute_zone_rmses,
    plot_diagnostics,
    plot_learning_curves,
    plot_results,
)


def _serialize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, torch.device):
            clean[k] = str(v)
        elif isinstance(v, tuple):
            clean[k] = list(v)
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def get_run_dir(base_dir: str, exp_name: str, dataset_name: str, seed: int, use_validation: bool = False) -> str:
    subdir = "runs_val" if use_validation else "runs"
    return os.path.join(base_dir, subdir, exp_name, dataset_name, f"seed_{seed}")


def is_valid_summary_file(summary_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        required_keys = {"exp_name", "dataset", "seed", "Test_RMSE", "Test_S_Score"}
        if required_keys.issubset(data.keys()) and data["Test_RMSE"] is not None:
            return data
    except Exception:
        return None
    return None


def find_existing_summary(
    exp_name: str,
    dataset_name: str,
    seed: int,
    out_dir: str,
    prev_results_dir: Optional[str] = None,
    use_validation: bool = False,
) -> Optional[Dict[str, Any]]:
    subdir = "runs_val" if use_validation else "runs"
    candidates = [
        os.path.join(get_run_dir(out_dir, exp_name, dataset_name, seed, use_validation=use_validation), "metrics_summary.json"),
    ]
    if prev_results_dir:
        candidates.append(
            os.path.join(
                get_run_dir(prev_results_dir, exp_name, dataset_name, seed, use_validation=use_validation),
                "metrics_summary.json",
            )
        )
        candidates.append(
            os.path.join(
                prev_results_dir,
                "results",
                subdir,
                exp_name,
                dataset_name,
                f"seed_{seed}",
                "metrics_summary.json",
            )
        )

    for path in candidates:
        valid_data = is_valid_summary_file(path)
        if valid_data is not None:
            return valid_data
    return None


def append_to_master_ledger(out_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    is_val = bool(summary.get("use_validation", False))
    suffix = "_val" if is_val else ""
    ledger_jsonl = os.path.join(out_dir, f"master_ledger{suffix}.jsonl")
    ledger_csv = os.path.join(out_dir, f"master_ledger{suffix}.csv")

    key = (summary["exp_name"], summary["dataset"], int(summary["seed"]))
    records_map: Dict[tuple, Dict[str, Any]] = {}

    if os.path.isfile(ledger_jsonl):
        with open(ledger_jsonl, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    r_key = (rec["exp_name"], rec["dataset"], int(rec["seed"]))
                    records_map[r_key] = rec
                except Exception:
                    continue

    ledger_entry = {k: v for k, v in summary.items() if k not in ("config", "epoch_history")}
    records_map[key] = ledger_entry

    sorted_records = sorted(
        records_map.values(),
        key=lambda r: (str(r.get("exp_name", "")), str(r.get("dataset", "")), int(r.get("seed", 0))),
    )

    tmp_jsonl = f"{ledger_jsonl}.tmp.{os.getpid()}"
    with open(tmp_jsonl, mode="w", encoding="utf-8") as f:
        for rec in sorted_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_jsonl, ledger_jsonl)

    csv_cols = [
        "exp_name",
        "exp_block",
        "dataset",
        "seed",
        "use_validation",
        "best_epoch",
        "val_raw_best_epoch",
        "val_smooth3_best_epoch",
        "val_smooth3_rmse",
        "val_selected_epoch",
        "Val_RMSE",
        "Val_S_Score",
        "Test_RMSE",
        "Test_S_Score",
        "oracle_best_epoch",
        "oracle_Test_RMSE",
        "oracle_Test_S_Score",
        "RMSE_0_20",
        "RMSE_20_50",
        "RMSE_50_100",
        "RMSE_100_150",
        "Median_PCC_Phys",
        "Median_PCC_RUL",
        "elapsed_sec",
        "timestamp",
    ]
    tmp_csv = f"{ledger_csv}.tmp.{os.getpid()}"
    with open(tmp_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for rec in sorted_records:
            writer.writerow(rec)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_csv, ledger_csv)


def duplicate_run_summary(
    source_summary: Dict[str, Any],
    target_exp_name: str,
    target_preset: Dict[str, Any],
    out_dir: str,
    prev_results_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    use_val = bool(source_summary.get("use_validation", False))
    src_exp_name = str(source_summary["exp_name"])
    dataset_name = str(source_summary["dataset"])
    seed = int(source_summary["seed"])

    dup = json.loads(json.dumps(source_summary))
    dup["exp_name"] = target_exp_name
    dup["exp_block"] = target_preset.get("block", "")
    dup["exp_title"] = target_preset.get("title", "")
    dup["exp_latex_name"] = target_preset.get("latex_name", target_exp_name)
    dup["duplicated_from"] = src_exp_name
    if "config" in dup and isinstance(dup["config"], dict):
        dup["config"]["exp_name"] = target_exp_name
        dup["config"]["exp_block"] = target_preset.get("block", "")
        dup["config"]["exp_title"] = target_preset.get("title", "")

    run_dir = get_run_dir(out_dir, target_exp_name, dataset_name, seed, use_validation=use_val)
    os.makedirs(run_dir, exist_ok=True)

    # Копируем также CSV, PNG и TensorBoard артефакты из исходной папки прогона
    candidate_src_dirs = [get_run_dir(out_dir, src_exp_name, dataset_name, seed, use_validation=use_val)]
    if prev_results_dirs:
        subdir = "runs_val" if use_val else "runs"
        for p_dir in prev_results_dirs:
            candidate_src_dirs.append(get_run_dir(p_dir, src_exp_name, dataset_name, seed, use_validation=use_val))
            candidate_src_dirs.append(os.path.join(p_dir, "results", subdir, src_exp_name, dataset_name, f"seed_{seed}"))

    import shutil
    for src_dir in candidate_src_dirs:
        if os.path.isdir(src_dir) and os.path.abspath(src_dir) != os.path.abspath(run_dir):
            for item in os.listdir(src_dir):
                if item == "metrics_summary.json" or item.endswith(".pth"):
                    continue
                s_item = os.path.join(src_dir, item)
                d_item = os.path.join(run_dir, item)
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                elif os.path.isfile(s_item):
                    shutil.copy2(s_item, d_item)
            break

    summary_path = os.path.join(run_dir, "metrics_summary.json")
    atomic_write_json(summary_path, dup)
    append_to_master_ledger(out_dir, dup)
    return dup


def train_and_evaluate_single_run(
    cfg: Dict[str, Any],
    out_dir: str,
    save_weights: bool = False,
    save_plots: bool = False,
    prev_results_dirs: Optional[List[str]] = None,
    target_epoch: Optional[int] = None,
    enable_tb: bool = True,
) -> Dict[str, Any]:
    start_time = time.time()
    exp_name = cfg.get("exp_name", "00_full_model")
    dataset_name = cfg["dataset_name"]
    seed = int(cfg["seed"])
    device = cfg["device"]
    use_val = bool(cfg.get("use_validation", False))

    run_dir = get_run_dir(out_dir, exp_name, dataset_name, seed, use_validation=use_val)
    os.makedirs(run_dir, exist_ok=True)

    val_select_mode = str(cfg.get("val_select_mode", "min")).lower()
    val_ref_epoch: Optional[int] = target_epoch
    if not use_val and val_ref_epoch is None:
        val_summary = find_existing_summary(
            exp_name, dataset_name, seed, out_dir=out_dir, prev_results_dir=None, use_validation=True
        )
        if val_summary is None and prev_results_dirs:
            for p_dir in prev_results_dirs:
                val_summary = find_existing_summary(
                    exp_name, dataset_name, seed, out_dir=out_dir, prev_results_dir=p_dir, use_validation=True
                )
                if val_summary is not None:
                    break
        if val_summary is not None:
            raw_ep = val_summary.get("val_raw_best_epoch") or val_summary.get("best_epoch")
            smooth_ep = val_summary.get("val_smooth3_best_epoch")
            if val_select_mode == "smooth3" and smooth_ep is not None:
                val_ref_epoch = int(smooth_ep)
            elif val_select_mode == "auto" and raw_ep is not None and smooth_ep is not None:
                # Если точечный минимум произошел слишком рано (отрыв > 2 эпох от центра окна 3 эпох), берем центр окна 3 эпох
                val_ref_epoch = int(smooth_ep) if (int(smooth_ep) - int(raw_ep) > 2) else int(raw_ep)
            elif raw_ep is not None:
                val_ref_epoch = int(raw_ep)

    mode_banner = "VALIDATION (85% Train / 15% Val)" if use_val else "FULL TRAIN (100% Train -> Test)"
    print(f"\n{'=' * 80}")
    print(f"[RUN] EXPERIMENT: {exp_name} | DATASET: {dataset_name} | SEED: {seed}")
    print(f"   Mode:  {mode_banner}")
    print(f"   Title: {cfg.get('exp_title', '')}")
    if not use_val and val_ref_epoch is not None:
        print(f"   [VAL REF] Target epoch from validation (85/15): Epoch {val_ref_epoch}")
    print(f"{'=' * 80}")

    set_seed(seed)

    train_loader, val_loader, X_train, y_train, X_test, y_test, features = prepare_dataloaders(cfg)

    encoder = build_model(cfg, n_sensors=len(features))
    criterion = build_criterion(cfg, encoder).to(device)

    optimizer = build_optimizer(encoder, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema_encoder = AveragedModel(encoder, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)

    csv_path = os.path.join(run_dir, "epoch_log.csv")
    best_model_path = os.path.join(run_dir, f"best_model_{dataset_name}_seed{seed}.pth")

    writer = None
    if enable_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    best_target_rmse = float("inf")
    best_test_rmse = float("inf")
    best_test_score = float("inf")
    best_val_rmse: Optional[float] = None
    best_val_score: Optional[float] = None
    best_epoch = 1
    best_preds: Optional[np.ndarray] = None
    best_module_state: Optional[Dict[str, torch.Tensor]] = None

    val_ref_preds: Optional[np.ndarray] = None
    val_ref_module_state: Optional[Dict[str, torch.Tensor]] = None
    val_ref_metrics: Optional[Dict[str, Any]] = None

    epoch_history: List[Dict[str, Any]] = []
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    save_epoch_cfg = cfg.get("save_epoch", None)
    save_epochs_set = set()
    if save_epoch_cfg is not None:
        if isinstance(save_epoch_cfg, int):
            save_epochs_set.add(save_epoch_cfg)
        elif isinstance(save_epoch_cfg, (list, tuple)):
            save_epochs_set.update(int(x) for x in save_epoch_cfg)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "Epoch",
                "Total_Loss",
                "order_Loss",
                "MSE_Loss",
                "MSE_Proxy_Loss",
                "KL_Loss",
                "Weighted_MSE_Loss",
                "Weighted_KL_Loss",
                "Weighted_order_Loss",
                "Val_RMSE",
                "Val_S_Score",
                "Test_RMSE",
                "Test_S_Score",
                "RMSE_0_20",
                "RMSE_20_50",
                "RMSE_50_100",
                "RMSE_100_150",
            ]
        )

        for epoch in range(cfg["epochs"]):
            encoder.train()
            epoch_loss = 0.0
            epoch_order = 0.0
            epoch_mse_loss = 0.0
            epoch_mse_proxy = 0.0
            epoch_kl = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                pred_rul, log_weights, emb_norm, proxies_norm = encoder(batch_x)

                loss, loss_mse_tensor, loss_kl, loss_order, loss_mse_proxy = criterion(
                    pred_rul,
                    log_weights,
                    batch_y,
                    proxies_norm=proxies_norm,
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer.step()
                ema_encoder.update_parameters(encoder)

                epoch_loss += loss.item()
                epoch_mse_loss += loss_mse_tensor.item()
                epoch_mse_proxy += loss_mse_proxy.item()
                epoch_kl += loss_kl.item()
                epoch_order += loss_order.item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            avg_order = epoch_order / len(train_loader)
            avg_mse_loss = epoch_mse_loss / len(train_loader)
            avg_mse_proxy = epoch_mse_proxy / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)

            w_mse = float(cfg.get("mse_weight", 0.05))
            w_kl = float(cfg.get("kl_weight", 1.0))
            w_ord = float(cfg.get("lambda_order", 0.5))
            if cfg.get("head_type", "proxy") == "proxy" and cfg.get("use_residual", True):
                weighted_mse = w_mse * avg_mse_proxy + w_mse * avg_mse_loss
            else:
                weighted_mse = w_mse * avg_mse_loss
            weighted_kl = w_kl * avg_kl
            weighted_order = w_ord * avg_order

            ema_encoder.eval()

            epoch_val_rmse: Optional[float] = None
            epoch_val_score: Optional[float] = None
            if use_val and val_loader is not None:
                val_preds, val_trues = [], []
                with torch.no_grad():
                    for v_x, v_y in val_loader:
                        v_x = v_x.to(device)
                        v_pred, _, _, _ = ema_encoder(v_x)
                        val_preds.extend(v_pred.cpu().numpy())
                        val_trues.extend(v_y.numpy())

                val_preds_arr = np.clip(np.array(val_preds), a_min=0, a_max=cfg["max_rul"])
                val_trues_arr = np.array(val_trues)
                epoch_val_rmse = float(np.sqrt(np.mean((val_preds_arr - val_trues_arr) ** 2)))
                dv = val_preds_arr - val_trues_arr
                epoch_val_score = float(np.sum(np.where(dv < 0, np.exp(-dv / 13.0) - 1, np.exp(dv / 10.0) - 1)))

            with torch.no_grad():
                test_preds, _, _, _ = ema_encoder(test_tensor)
                preds = test_preds.cpu().numpy()

            preds = np.clip(preds, a_min=0, a_max=cfg["max_rul"])

            epoch_test_rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
            d = preds - y_test
            epoch_score = float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))
            epoch_zones = compute_zone_rmses(y_test, preds)

            ep_num = epoch + 1
            ep_record = {
                "epoch": ep_num,
                "Total_Loss": round(avg_loss, 4),
                "order_Loss": round(avg_order, 4),
                "MSE_Loss": round(avg_mse_loss, 4),
                "MSE_Proxy_Loss": round(avg_mse_proxy, 4),
                "KL_Loss": round(avg_kl, 4),
                "Weighted_MSE_Loss": round(weighted_mse, 4),
                "Weighted_KL_Loss": round(weighted_kl, 4),
                "Weighted_order_Loss": round(weighted_order, 4),
                "Val_RMSE": round(epoch_val_rmse, 4) if epoch_val_rmse is not None else None,
                "Val_S_Score": round(epoch_val_score, 2) if epoch_val_score is not None else None,
                "Test_RMSE": round(epoch_test_rmse, 4),
                "Test_S_Score": round(epoch_score, 2),
                **epoch_zones,
            }
            epoch_history.append(ep_record)

            if writer is not None:
                writer.add_scalar("Loss/Train_Total", avg_loss, epoch)
                writer.add_scalar("Loss/Train_order", avg_order, epoch)
                writer.add_scalar("Loss/Train_MSE", avg_mse_loss, epoch)
                writer.add_scalar("Loss/Train_MSE_Proxy", avg_mse_proxy, epoch)
                writer.add_scalar("Loss/Train_KL", avg_kl, epoch)
                writer.add_scalar("Loss_Weighted/MSE", weighted_mse, epoch)
                writer.add_scalar("Loss_Weighted/KL", weighted_kl, epoch)
                writer.add_scalar("Loss_Weighted/Gram_Order", weighted_order, epoch)
                if epoch_val_rmse is not None:
                    writer.add_scalar("Metric/Val_RMSE", epoch_val_rmse, epoch)
                if epoch_val_score is not None:
                    writer.add_scalar("Metric/Val_S_Score", epoch_val_score, epoch)
                writer.add_scalar("Metric/Test_RMSE", epoch_test_rmse, epoch)
                writer.add_scalar("Metric/Test_S_Score", epoch_score, epoch)
                writer.add_scalar("Zone_RMSE/0_20_Critical", epoch_zones["RMSE_0_20"], epoch)
                writer.add_scalar("Zone_RMSE/20_50_Danger", epoch_zones["RMSE_20_50"], epoch)
                writer.add_scalar("Zone_RMSE/50_100_Transition", epoch_zones["RMSE_50_100"], epoch)
                writer.add_scalar("Zone_RMSE/100_150_Normal", epoch_zones["RMSE_100_150"], epoch)
                writer.add_scalar("Hyperparam/Learning_Rate", scheduler.get_last_lr()[0], epoch)

            val_log_str = (
                f" | Val RMSE: {epoch_val_rmse:.2f} (Val S: {epoch_val_score:.1f})"
                if epoch_val_rmse is not None and epoch_val_score is not None
                else ""
            )
            val_csv_rmse = round(epoch_val_rmse, 4) if epoch_val_rmse is not None else ""
            val_csv_score = round(epoch_val_score, 2) if epoch_val_score is not None else ""
            ref_marker = "  <-- [VAL EPOCH]" if (not use_val and val_ref_epoch == ep_num) else ""

            print(
                f"[{exp_name} | {dataset_name} | s={seed}] Epoch [{ep_num:02d}/{cfg['epochs']}] | "
                f"Loss: {avg_loss:.3f} | MSE_L: {avg_mse_loss:.3f} | KL: {avg_kl:.3f} | Gram: {avg_order:.4f}{val_log_str} | "
                f"Test RMSE: {epoch_test_rmse:.2f} | S-Score: {epoch_score:.2f}{ref_marker}"
            )

            csv_writer.writerow(
                [
                    ep_num,
                    round(avg_loss, 4),
                    round(avg_order, 4),
                    round(avg_mse_loss, 4),
                    round(avg_mse_proxy, 4),
                    round(avg_kl, 4),
                    round(weighted_mse, 4),
                    round(weighted_kl, 4),
                    round(weighted_order, 4),
                    val_csv_rmse,
                    val_csv_score,
                    round(epoch_test_rmse, 4),
                    round(epoch_score, 2),
                    epoch_zones["RMSE_0_20"],
                    epoch_zones["RMSE_20_50"],
                    epoch_zones["RMSE_50_100"],
                    epoch_zones["RMSE_100_150"],
                ]
            )
            csv_file.flush()

            if not use_val and val_ref_epoch is not None and ep_num == val_ref_epoch:
                val_ref_preds = preds.copy()
                val_ref_module_state = {
                    k: v.detach().cpu().clone() for k, v in ema_encoder.module.state_dict().items()
                }
                val_ref_metrics = ep_record.copy()

            if ep_num in save_epochs_set:
                ep_ckpt_path = os.path.join(run_dir, f"epoch{ep_num}_model_{dataset_name}_seed{seed}.pth")
                torch.save(ema_encoder.module.state_dict(), ep_ckpt_path)

            target_metric = (
                epoch_val_rmse if (use_val and epoch_val_rmse is not None) else epoch_test_rmse
            )

            if target_metric < best_target_rmse:
                best_target_rmse = target_metric
                best_test_rmse = epoch_test_rmse
                best_test_score = epoch_score
                best_val_rmse = epoch_val_rmse
                best_val_score = epoch_val_score
                best_epoch = ep_num
                best_preds = preds.copy()
                best_module_state = {
                    k: v.detach().cpu().clone() for k, v in ema_encoder.module.state_dict().items()
                }
                if save_weights and (use_val or val_ref_epoch is None):
                    torch.save(best_module_state, best_model_path)
                if writer is not None:
                    writer.add_scalar("Best_Epoch/Test_RMSE", epoch_test_rmse, epoch)
                    writer.add_scalar("Best_Epoch/Test_S_Score", epoch_score, epoch)
                    writer.add_histogram("Best_Epoch/Test_Predictions_Dist", preds, epoch)
                    writer.add_histogram("Best_Epoch/Proxy_RUL_Activations", test_preds.cpu(), epoch)

    # На этапе валидации (85/15) вычисляем как точечный минимум Val_RMSE, так и сглаженный по окну из 3 эпох
    # (защита от случайных ранних провалов / спайков валидационной кривой)
    val_raw_best_epoch: Optional[int] = best_epoch if use_val else None
    val_smooth3_best_epoch: Optional[int] = None
    val_smooth3_rmse: Optional[float] = None

    if use_val and len(epoch_history) >= 3:
        val_rmses = [
            float(ep["Val_RMSE"]) if ep.get("Val_RMSE") is not None else float("inf")
            for ep in epoch_history
        ]
        best_win_avg = float("inf")
        best_center_idx = 0
        for idx_c in range(1, len(val_rmses) - 1):
            win_avg = (val_rmses[idx_c - 1] + val_rmses[idx_c] + val_rmses[idx_c + 1]) / 3.0
            if win_avg < best_win_avg:
                best_win_avg = win_avg
                best_center_idx = idx_c
        val_smooth3_best_epoch = int(epoch_history[best_center_idx]["epoch"])
        val_smooth3_rmse = round(best_win_avg, 4)

    if use_val and val_select_mode == "smooth3" and val_smooth3_best_epoch is not None:
        chosen_epoch = val_smooth3_best_epoch
    elif (
        use_val
        and val_select_mode == "auto"
        and val_raw_best_epoch is not None
        and val_smooth3_best_epoch is not None
        and (val_smooth3_best_epoch - val_raw_best_epoch > 2)
    ):
        chosen_epoch = val_smooth3_best_epoch
    elif not use_val and val_ref_metrics is not None and val_ref_preds is not None and val_ref_module_state is not None:
        chosen_epoch = int(val_ref_epoch)
        chosen_test_rmse = float(val_ref_metrics["Test_RMSE"])
        chosen_test_score = float(val_ref_metrics["Test_S_Score"])
        chosen_preds = val_ref_preds
        chosen_module_state = val_ref_module_state
        if save_weights:
            torch.save(chosen_module_state, best_model_path)
    else:
        chosen_epoch = best_epoch

    if use_val or val_ref_metrics is None or val_ref_preds is None or val_ref_module_state is None:
        chosen_test_rmse = best_test_rmse
        chosen_test_score = best_test_score
        chosen_preds = best_preds
        chosen_module_state = best_module_state

    if use_val:
        smooth_str = (
            f" | Smooth-3 Center Epoch: {val_smooth3_best_epoch:02d} (3-ep avg Val RMSE: {val_smooth3_rmse:.2f})"
            if val_smooth3_best_epoch is not None and val_smooth3_rmse is not None
            else ""
        )
        print(
            f"\n[VALIDATION 85/15 DONE | {exp_name} | {dataset_name} | seed={seed}]\n"
            f"   -> BEST VAL EPOCH (Min): Epoch {best_epoch:02d}/{cfg['epochs']} | "
            f"Min Val RMSE: {best_val_rmse:.2f} | Val S-Score: {best_val_score:.2f}{smooth_str}"
        )
    else:
        print(f"\n[100% TRAIN DONE | {exp_name} | {dataset_name} | seed={seed}]")
        if val_ref_metrics is not None:
            print(
                f"   -> At validation epoch (Epoch {chosen_epoch:02d}): "
                f"Test RMSE = {chosen_test_rmse:.2f} | S-Score = {chosen_test_score:.2f}"
            )
        print(
            f"   -> Best epoch on test (Epoch {best_epoch:02d}): "
            f"Test RMSE = {best_test_rmse:.2f} | S-Score = {best_test_score:.2f}"
        )

    if chosen_module_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in chosen_module_state.items()})
    encoder.eval()

    assert chosen_preds is not None
    zone_rmses = compute_zone_rmses(y_test, chosen_preds)

    # Сохраняем по-двигательную таблицу предсказаний для воспроизводимой проверки рецензентами
    preds_csv_path = os.path.join(run_dir, f"test_predictions_{dataset_name}_seed{seed}.csv")
    with open(preds_csv_path, mode="w", newline="", encoding="utf-8") as pf:
        p_writer = csv.writer(pf)
        p_writer.writerow(["Unit_Index", "True_RUL", "Pred_RUL", "Error_Pred_minus_True", "Abs_Error", "Sq_Error"])
        for idx_u, (yt, yp) in enumerate(zip(y_test, chosen_preds), start=1):
            err = float(yp - yt)
            p_writer.writerow([idx_u, round(float(yt), 4), round(float(yp), 4), round(err, 4), round(abs(err), 4), round(err ** 2, 4)])

    eval_train_loader = DataLoader(
        CMAPSSDataset(X_train, y_train), batch_size=cfg["batch_size"], shuffle=False
    )
    med_pcc_phys, med_pcc_rul, _, _ = compute_smart_pccs(encoder, eval_train_loader, cfg, max_samples=1000)

    if writer is not None:
        writer.add_scalar("Best_Epoch/Selected_Epoch", chosen_epoch, cfg["epochs"])
        writer.add_scalar("Best_Epoch/Final_Test_RMSE", chosen_test_rmse, cfg["epochs"])
        writer.add_scalar("Best_Epoch/Final_Test_S_Score", chosen_test_score, cfg["epochs"])
        writer.add_scalar("Best_Epoch/Median_PCC_Phys", med_pcc_phys, cfg["epochs"])
        writer.add_scalar("Best_Epoch/Median_PCC_RUL", med_pcc_rul, cfg["epochs"])

    if save_plots:
        print(f"[{dataset_name}] Saving learning curves, proxy diagnostics, and results plot...")
        fig_lc = plot_learning_curves(
            csv_path,
            dataset_name,
            run_dir,
            mse_weight=float(cfg.get("mse_weight", 0.05)),
            kl_weight=float(cfg.get("kl_weight", 1.0)),
            lambda_order=float(cfg.get("lambda_order", 0.5)),
        )
        if writer is not None and fig_lc is not None:
            writer.add_figure("Best_Epoch/Learning_Curves", fig_lc, global_step=cfg["epochs"])
        if fig_lc is not None:
            plt.close(fig_lc)

        # Диагностика вызывается после цикла обучения на детерминированном eval_train_loader
        fig_diag = plot_diagnostics(
            encoder, criterion, eval_train_loader, cfg, dataset_name, chosen_epoch - 1, save_dir=run_dir
        )
        if writer is not None and fig_diag is not None:
            writer.add_figure("Best_Epoch/Proxy_Activation_Histogram", fig_diag, global_step=cfg["epochs"])
        if fig_diag is not None:
            plt.close(fig_diag)

        proxies_norm_np: Optional[np.ndarray] = None
        proxy_ruls_np: Optional[np.ndarray] = None
        if getattr(encoder, "proxies", None) is not None and getattr(encoder, "proxy_ruls", None) is not None:
            with torch.no_grad():
                proxies_norm_np = F.normalize(encoder.proxies, p=2, dim=1).cpu().numpy()
                proxy_ruls_np = encoder.proxy_ruls.cpu().numpy()

        train_emb_list = []
        with torch.no_grad():
            for batch_x, _ in eval_train_loader:
                _, _, emb, _ = encoder(batch_x.to(device))
                train_emb_list.append(emb.cpu().numpy())

        train_embeddings = np.vstack(train_emb_list)
        sample_size = min(3000, len(train_embeddings))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(train_embeddings), sample_size, replace=False)

        fig_res = plot_results(
            y_true=y_test,
            y_pred=chosen_preds,
            train_emb=train_embeddings[indices],
            y_train_subset=y_train[indices],
            proxies_emb=proxies_norm_np,
            proxy_ruls=proxy_ruls_np,
            dataset_name=dataset_name,
            save_dir=run_dir,
        )
        if writer is not None and fig_res is not None:
            writer.add_figure("Best_Epoch/Metric_Learning_Topology", fig_res, global_step=cfg["epochs"])
        if fig_res is not None:
            plt.close(fig_res)

    if writer is not None:
        writer.flush()
        writer.close()

    elapsed_sec = round(time.time() - start_time, 2)
    summary: Dict[str, Any] = {
        "exp_name": exp_name,
        "exp_block": cfg.get("exp_block", ""),
        "exp_title": cfg.get("exp_title", ""),
        "exp_latex_name": cfg.get("exp_latex_name", exp_name),
        "dataset": dataset_name,
        "seed": seed,
        "use_validation": use_val,
        "best_epoch": chosen_epoch,
        "val_raw_best_epoch": val_raw_best_epoch,
        "val_smooth3_best_epoch": val_smooth3_best_epoch,
        "val_smooth3_rmse": val_smooth3_rmse,
        "val_selected_epoch": val_ref_epoch,
        "Val_RMSE": round(best_val_rmse, 4) if best_val_rmse is not None else None,
        "Val_S_Score": round(best_val_score, 2) if best_val_score is not None else None,
        "Test_RMSE": round(chosen_test_rmse, 4),
        "Test_S_Score": round(chosen_test_score, 2),
        "oracle_best_epoch": best_epoch,
        "oracle_Test_RMSE": round(best_test_rmse, 4),
        "oracle_Test_S_Score": round(best_test_score, 2),
        "RMSE_0_20": zone_rmses["RMSE_0_20"],
        "RMSE_20_50": zone_rmses["RMSE_20_50"],
        "RMSE_50_100": zone_rmses["RMSE_50_100"],
        "RMSE_100_150": zone_rmses["RMSE_100_150"],
        "Median_PCC_Phys": med_pcc_phys,
        "Median_PCC_RUL": med_pcc_rul,
        "elapsed_sec": elapsed_sec,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch_history": epoch_history,
        "config": _serialize_config(cfg),
    }

    summary_path = os.path.join(run_dir, "metrics_summary.json")
    atomic_write_json(summary_path, summary)
    append_to_master_ledger(out_dir, summary)

    return summary
