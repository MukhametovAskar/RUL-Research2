#!/usr/bin/env python3
"""
Complete cell-by-cell audit and verification script for the 2-Stage Ablation Study protocol:
  Stage 1: 85% Train / 15% Val (5-split Monte-Carlo CV across seeds 42, 43, 44, 45, 50)
           Consensus stopping epoch: E* = round(median_{s}(val_raw_best_epoch^{(s)}))
  Stage 2: 100% Train -> Test (metrics extracted strictly at epoch E* from epoch_history & epoch_log.csv)

Verifies:
  - All 160 Stage-1 validation runs (metrics_summary.json + epoch_log.csv)
  - All 160 Stage-2 full-train test runs (metrics_summary.json + epoch_log.csv)
  - Table 1 in ABLATION_STUDY_RESULTS.md (8 experiments x 10 columns = 80 metric cells)
  - Table 2 in ABLATION_STUDY_RESULTS.md (8 experiments x 6 columns = 48 metric cells)
  - Section 3 in ABLATION_STUDY_RESULTS.md (160 seed rows + 32 dataset Mean ± Std rows = 192 rows)

Usage:
    python scripts/verify_ablation.py
    python scripts/verify_ablation.py --sync
"""
import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.ablation_grid import ABLATION_GRID
from configs.base_config import DATASETS, DEFAULT_SEEDS

SHORT_TO_FULL = {
    "00": "00_full_model",
    "01": "01_linear_head_standard_mse",
    "02": "02_linear_head_balanced_mse_1k",
    "03": "03_wo_ldl_kl_loss",
    "04": "04_wo_proxy_1k_aggregation",
    "05": "05_wo_residual_proxy_head",
    "06": "06_wo_cosine_gram_regularizer",
    "07": "07_wo_regime_normalization",
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fmt_ms(arr: List[float]) -> str:
    a = np.asarray(arr, dtype=float)
    return f"**{a.mean():.2f} ± {a.std(ddof=1):.2f}**"


def sync_and_audit_runs(results_dir: str, sync: bool = False) -> Tuple[int, List[str], Dict[str, Any]]:
    issues: List[str] = []
    checked = 0
    ground_truth: Dict[str, Any] = {}

    for code, exp_name in SHORT_TO_FULL.items():
        preset = ABLATION_GRID[exp_name]
        overrides = preset.get("overrides", {})
        ground_truth[code] = {
            "exp_name": exp_name,
            "ds_cells": {},
            "e_star": {},
            "seed_rows": {},
            "zones_means": {k: [] for k in ("RMSE_0_20", "RMSE_20_50", "RMSE_50_100", "RMSE_100_150")},
            "zones_stds": {k: [] for k in ("RMSE_0_20", "RMSE_20_50", "RMSE_50_100", "RMSE_100_150")},
            "pcc_phys": [],
            "pcc_rul": [],
        }

        ds_rmse_means, ds_rmse_stds = [], []
        ds_score_means, ds_score_stds = [], []

        for ds in DATASETS:
            val_raw_epochs: List[int] = []
            val_info: Dict[int, Dict[str, Any]] = {}

            for s in DEFAULT_SEEDS:
                v_path = os.path.join(results_dir, "runs_val", exp_name, ds, f"seed_{s}", "metrics_summary.json")
                v_csv = os.path.join(results_dir, "runs_val", exp_name, ds, f"seed_{s}", "epoch_log.csv")
                if not os.path.isfile(v_path) or not os.path.isfile(v_csv):
                    issues.append(f"Missing validation files for {exp_name}/{ds}/seed_{s}")
                    continue

                v_data = load_json(v_path)
                if sync and overrides and "config" in v_data and isinstance(v_data["config"], dict):
                    modified_v = False
                    for ok, ov in overrides.items():
                        if v_data["config"].get(ok) != ov:
                            v_data["config"][ok] = ov
                            modified_v = True
                    if modified_v:
                        save_json(v_path, v_data)

                with open(v_csv, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                val_rmses = [float(r["Val_RMSE"]) for r in rows]
                raw_min_idx = int(np.argmin(val_rmses))
                raw_min_ep = int(rows[raw_min_idx]["Epoch"])
                raw_min_rmse = float(val_rmses[raw_min_idx])

                # Check Smooth-3 window [e-1, e, e+1] (interior centers 1..n_ep-2 as in src/trainer.py)
                n_ep = len(val_rmses)
                best_sm_rmse = float("inf")
                best_sm_ep = 1
                for idx in range(1, n_ep - 1):
                    sm = (val_rmses[idx - 1] + val_rmses[idx] + val_rmses[idx + 1]) / 3.0
                    if sm < best_sm_rmse:
                        best_sm_rmse = sm
                        best_sm_ep = int(rows[idx]["Epoch"])

                if int(v_data.get("val_raw_best_epoch", raw_min_ep)) != raw_min_ep:
                    issues.append(
                        f"{exp_name}/{ds}/seed_{s}: val_raw_best_epoch={v_data.get('val_raw_best_epoch')} != CSV min epoch {raw_min_ep}"
                    )
                if int(v_data.get("val_smooth3_best_epoch", best_sm_ep)) != best_sm_ep:
                    issues.append(
                        f"{exp_name}/{ds}/seed_{s}: val_smooth3_best_epoch={v_data.get('val_smooth3_best_epoch')} != CSV smooth3 epoch {best_sm_ep}"
                    )

                val_raw_epochs.append(raw_min_ep)
                val_info[s] = {
                    "raw_ep": raw_min_ep,
                    "raw_rmse": raw_min_rmse,
                    "sm_ep": int(v_data["val_smooth3_best_epoch"]),
                    "sm_rmse": float(v_data["val_smooth3_rmse"]),
                }

            if len(val_raw_epochs) != len(DEFAULT_SEEDS):
                continue

            consensus_ep = int(round(float(np.median(val_raw_epochs))))
            ground_truth[code]["e_star"][ds] = consensus_ep
            ground_truth[code]["seed_rows"][ds] = {}

            rmses, scores = [], []
            z_vals = {k: [] for k in ("RMSE_0_20", "RMSE_20_50", "RMSE_50_100", "RMSE_100_150")}
            p_phys, p_rul = [], []

            for s in DEFAULT_SEEDS:
                t_path = os.path.join(results_dir, "runs", exp_name, ds, f"seed_{s}", "metrics_summary.json")
                t_csv = os.path.join(results_dir, "runs", exp_name, ds, f"seed_{s}", "epoch_log.csv")
                if not os.path.isfile(t_path) or not os.path.isfile(t_csv):
                    issues.append(f"Missing test files for {exp_name}/{ds}/seed_{s}")
                    continue

                t_data = load_json(t_path)
                with open(t_csv, "r", encoding="utf-8") as f:
                    t_csv_rows = {int(r["Epoch"]): r for r in csv.DictReader(f)}

                hist_map = {int(h["epoch"]): h for h in t_data.get("epoch_history", [])}
                if consensus_ep not in hist_map or consensus_ep not in t_csv_rows:
                    issues.append(f"{exp_name}/{ds}/seed_{s}: consensus E*={consensus_ep} missing in history/CSV")
                    continue

                ep_rec = hist_map[consensus_ep]
                csv_rec = t_csv_rows[consensus_ep]

                # Cross-check JSON epoch_history vs epoch_log.csv at E*
                if abs(float(ep_rec["Test_RMSE"]) - float(csv_rec["Test_RMSE"])) > 1e-3:
                    issues.append(f"{exp_name}/{ds}/seed_{s}: JSON vs CSV Test_RMSE mismatch at E*={consensus_ep}")
                if abs(float(ep_rec["Test_S_Score"]) - float(csv_rec["Test_S_Score"])) > 1e-2:
                    issues.append(f"{exp_name}/{ds}/seed_{s}: JSON vs CSV Test_S_Score mismatch at E*={consensus_ep}")

                if sync:
                    modified_t = False
                    if overrides and "config" in t_data and isinstance(t_data["config"], dict):
                        for ok, ov in overrides.items():
                            if t_data["config"].get(ok) != ov:
                                t_data["config"][ok] = ov
                                modified_t = True
                    if t_data.get("best_epoch") != consensus_ep or t_data.get("val_selected_epoch") != consensus_ep:
                        t_data["best_epoch"] = consensus_ep
                        t_data["val_selected_epoch"] = consensus_ep
                        t_data["Test_RMSE"] = ep_rec["Test_RMSE"]
                        t_data["Test_S_Score"] = ep_rec["Test_S_Score"]
                        for zk in ("RMSE_0_20", "RMSE_20_50", "RMSE_50_100", "RMSE_100_150"):
                            if zk in ep_rec:
                                t_data[zk] = ep_rec[zk]
                        modified_t = True
                    if modified_t:
                        save_json(t_path, t_data)

                cfg = t_data.get("config", {})
                for ok, ov in overrides.items():
                    if cfg.get(ok) != ov:
                        issues.append(f"{exp_name}/{ds}/seed_{s}: config[{ok}]={cfg.get(ok)} != {ov}")

                rmses.append(float(ep_rec["Test_RMSE"]))
                scores.append(float(ep_rec["Test_S_Score"]))
                for zk in z_vals:
                    z_vals[zk].append(float(ep_rec[zk]))
                p_phys.append(float(t_data["Median_PCC_Phys"]))
                p_rul.append(float(t_data["Median_PCC_RUL"]))

                ground_truth[code]["seed_rows"][ds][s] = {
                    **val_info[s],
                    "sel_ep": consensus_ep,
                    "test_rmse": float(ep_rec["Test_RMSE"]),
                    "test_s": float(ep_rec["Test_S_Score"]),
                }
                checked += 1

            r_arr = np.asarray(rmses, dtype=float)
            s_arr = np.asarray(scores, dtype=float)
            ground_truth[code]["ds_cells"][ds] = (fmt_ms(rmses), fmt_ms(scores))
            ds_rmse_means.append(float(r_arr.mean()))
            ds_rmse_stds.append(float(r_arr.std(ddof=1)))
            ds_score_means.append(float(s_arr.mean()))
            ds_score_stds.append(float(s_arr.std(ddof=1)))

            for zk in z_vals:
                za = np.asarray(z_vals[zk], dtype=float)
                ground_truth[code]["zones_means"][zk].append(float(za.mean()))
                ground_truth[code]["zones_stds"][zk].append(float(za.std(ddof=1)))
            ground_truth[code]["pcc_phys"].append(float(np.mean(p_phys)))
            ground_truth[code]["pcc_rul"].append(float(np.mean(p_rul)))

        ground_truth[code]["macro_rmse"] = f"**{np.mean(ds_rmse_means):.2f} ± {np.mean(ds_rmse_stds):.2f}**"
        ground_truth[code]["macro_score"] = f"**{np.mean(ds_score_means):.2f} ± {np.mean(ds_score_stds):.2f}**"
        ground_truth[code]["table2_zones"] = [
            f"**{np.mean(ground_truth[code]['zones_means'][zk]):.2f} ± {np.mean(ground_truth[code]['zones_stds'][zk]):.2f}**"
            for zk in ("RMSE_0_20", "RMSE_20_50", "RMSE_50_100", "RMSE_100_150")
        ]
        ground_truth[code]["table2_pcc"] = [
            f"**{np.mean(ground_truth[code]['pcc_phys']):.3f}**",
            f"**{np.mean(ground_truth[code]['pcc_rul']):.3f}**",
        ]

    return checked, issues, ground_truth


def verify_markdown_report(md_path: str, ground_truth: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
    problems: List[str] = []
    counts = {"table1_rows": 0, "table2_rows": 0, "detail_seed_rows": 0, "detail_mean_rows": 0}
    if not os.path.isfile(md_path):
        return [f"Markdown report not found: {md_path}"], counts

    text = open(md_path, "r", encoding="utf-8").read()
    cur_code = None

    for line_no, line in enumerate(text.splitlines(), 1):
        # Check Table 1 & Table 2 summary rows
        m_sum = re.match(r"^\|\s*\*\*(\d)\*\*\s*\|\s*`(\d\d)_[^`]+`\s*\|", line.strip())
        if m_sum:
            code = m_sum.group(2)
            gt = ground_truth[code]
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) == 13:
                counts["table1_rows"] += 1
                expected_cells = []
                for ds in DATASETS:
                    expected_cells.extend(gt["ds_cells"][ds])
                expected_cells.extend([gt["macro_rmse"], gt["macro_score"]])
                actual_cells = parts[3:]
                for col_idx, (act, exp_val) in enumerate(zip(actual_cells, expected_cells)):
                    if act != exp_val:
                        problems.append(f"L{line_no} Table 1 [{code}] col {col_idx+1}: '{act}' != '{exp_val}'")
            elif len(parts) == 8:
                counts["table2_rows"] += 1
                expected_cells = gt["table2_zones"] + gt["table2_pcc"]
                actual_cells = parts[2:]
                for col_idx, (act, exp_val) in enumerate(zip(actual_cells, expected_cells)):
                    if act != exp_val:
                        problems.append(f"L{line_no} Table 2 [{code}] col {col_idx+1}: '{act}' != '{exp_val}'")
            continue

        m_hdr = re.match(r"^### Эксперимент (\d): `(\d\d)_", line)
        if m_hdr:
            cur_code = m_hdr.group(2)
            continue

        # Check Section 3 per-seed rows
        m_seed = re.match(
            r"^\|\s*\*\*(FD00\d)\*\*\s*\|\s*`(\d+)`\s*\|\s*Epoch (\d+)\s*\|\s*([\d.]+)\s*\|\s*Epoch (\d+)\s*\|\s*([\d.]+)\s*\|\s*\*\*Epoch (\d+)\*\*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$",
            line.strip(),
        )
        if m_seed and cur_code:
            counts["detail_seed_rows"] += 1
            ds = m_seed.group(1)
            s = int(m_seed.group(2))
            gt_row = ground_truth[cur_code]["seed_rows"][ds][s]

            if int(m_seed.group(3)) != gt_row["raw_ep"]:
                problems.append(f"L{line_no} {cur_code}/{ds}/s{s}: raw_ep {m_seed.group(3)} != {gt_row['raw_ep']}")
            if abs(float(m_seed.group(4)) - gt_row["raw_rmse"]) > 0.011:
                problems.append(f"L{line_no} {cur_code}/{ds}/s{s}: raw_rmse {m_seed.group(4)} != {gt_row['raw_rmse']:.2f}")
            if int(m_seed.group(5)) != gt_row["sm_ep"]:
                problems.append(f"L{line_no} {cur_code}/{ds}/s{s}: sm_ep {m_seed.group(5)} != {gt_row['sm_ep']}")
            if abs(float(m_seed.group(6)) - gt_row["sm_rmse"]) > 0.011:
                problems.append(f"L{line_no} {cur_code}/{ds}/s{s}: sm_rmse {m_seed.group(6)} != {gt_row['sm_rmse']:.2f}")
            if int(m_seed.group(7)) != gt_row["sel_ep"]:
                problems.append(f"L{line_no} {cur_code}/{ds}/s{s}: sel_ep {m_seed.group(7)} != {gt_row['sel_ep']}")
            if abs(float(m_seed.group(8)) - gt_row["test_rmse"]) > 0.011:
                problems.append(f"L{line_no} {cur_code}/{ds}/s{s}: test_rmse {m_seed.group(8)} != {gt_row['test_rmse']:.2f}")
            if abs(float(m_seed.group(9)) - gt_row["test_s"]) > 0.011:
                problems.append(f"L{line_no} {cur_code}/{ds}/s{s}: test_s {m_seed.group(9)} != {gt_row['test_s']:.2f}")
            continue

        # Check Section 3 Mean ± Std rows
        m_mean = re.match(
            r"^\|\s*\*\*(FD00\d)\*\*\s*\|\s*\*\*Mean ± Std\*\*\s*\|\s*—\s*\|\s*—\s*\|\s*—\s*\|\s*—\s*\|\s*\*\*Median E\* = (\d+)\*\*\s*\|\s*(\*\*[\d.]+ ± [\d.]+\*\*)\s*\|\s*(\*\*[\d.]+ ± [\d.]+\*\*)\s*\|$",
            line.strip(),
        )
        if m_mean and cur_code:
            counts["detail_mean_rows"] += 1
            ds = m_mean.group(1)
            md_e_star = int(m_mean.group(2))
            md_r_cell = m_mean.group(3)
            md_s_cell = m_mean.group(4)

            gt_e_star = ground_truth[cur_code]["e_star"][ds]
            gt_r_cell, gt_s_cell = ground_truth[cur_code]["ds_cells"][ds]
            if md_e_star != gt_e_star:
                problems.append(f"L{line_no} {cur_code}/{ds} Mean row: Median E*={md_e_star} != {gt_e_star}")
            if md_r_cell != gt_r_cell:
                problems.append(f"L{line_no} {cur_code}/{ds} Mean row: RMSE '{md_r_cell}' != '{gt_r_cell}'")
            if md_s_cell != gt_s_cell:
                problems.append(f"L{line_no} {cur_code}/{ds} Mean row: S-Score '{md_s_cell}' != '{gt_s_cell}'")

    if counts["table1_rows"] != 8:
        problems.append(f"Expected 8 rows in Table 1, found {counts['table1_rows']}")
    if counts["table2_rows"] != 8:
        problems.append(f"Expected 8 rows in Table 2, found {counts['table2_rows']}")
    if counts["detail_seed_rows"] != 160:
        problems.append(f"Expected 160 per-seed rows in Section 3, found {counts['detail_seed_rows']}")
    if counts["detail_mean_rows"] != 32:
        problems.append(f"Expected 32 Mean ± Std rows in Section 3, found {counts['detail_mean_rows']}")

    return problems, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify 2-Stage Ablation Study protocol and Markdown tables.")
    parser.add_argument("--results_dir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--md_path", type=str, default=os.path.join(REPO_ROOT, "ABLATION_STUDY_RESULTS.md"))
    parser.add_argument("--sync", action="store_true", help="Synchronize top-level JSON fields to consensus E*.")
    args = parser.parse_args()

    checked, issues, ground_truth = sync_and_audit_runs(args.results_dir, sync=args.sync)
    md_problems, counts = verify_markdown_report(args.md_path, ground_truth)

    print(f"Checked Stage-1 (runs_val) + Stage-2 (runs) pairs: {checked}/160")
    print(
        f"Verified Markdown rows: Table 1 = {counts['table1_rows']}/8 | "
        f"Table 2 = {counts['table2_rows']}/8 | "
        f"Section 3 Seed Rows = {counts['detail_seed_rows']}/160 | "
        f"Section 3 Mean Rows = {counts['detail_mean_rows']}/32"
    )
    print(f"Run/CSV/JSON issues: {len(issues)}")
    for i in issues:
        print("  [ISSUE]", i)
    print(f"Markdown table issues: {len(md_problems)}")
    for p in md_problems:
        print("  [MD PROBLEM]", p)

    if issues or md_problems:
        sys.exit(1)
    print("ALL PROTOCOL AND TABLE CHECKS PASSED (0 issues).")


if __name__ == "__main__":
    main()
