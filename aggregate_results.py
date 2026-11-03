import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.ablation_grid import ABLATION_GRID, ALL_PRESETS
from configs.base_config import DATASETS, DEFAULT_SEEDS
from src.trainer import is_valid_summary_file


def discover_summaries_by_mode(
    out_dir: str, prev_results_dirs: Optional[List[str]] = None
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Dict[Tuple[str, str, int], Dict[str, Any]]]:
    full_train_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    val_runs_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    search_dirs: List[str] = []
    if prev_results_dirs:
        for p_dir in prev_results_dirs:
            if p_dir and os.path.isdir(p_dir):
                search_dirs.append(p_dir)
    if out_dir and os.path.isdir(out_dir):
        search_dirs.append(out_dir)

    for base in search_dirs:
        for root, _, files in os.walk(base):
            if "metrics_summary.json" in files:
                fpath = os.path.join(root, "metrics_summary.json")
                data = is_valid_summary_file(fpath)
                if data is not None:
                    key = (str(data["exp_name"]), str(data["dataset"]).upper(), int(data["seed"]))
                    is_val = bool(data.get("use_validation", False)) or ("runs_val" in root.replace("\\", "/"))
                    if is_val:
                        val_runs_map[key] = data
                    else:
                        full_train_map[key] = data

    # Для 07_wo_regime_normalization на однорежимных FD001/FD003 автоматически подставляем результаты 00_full_model,
    # если они еще не скопированы явно
    for exp_name, preset in ALL_PRESETS.items():
        only_ds = preset.get("only_datasets")
        dup_from = preset.get("duplicate_from")
        if only_ds and dup_from:
            for (src_exp, ds, s), src_data in list(full_train_map.items()):
                if src_exp == dup_from and ds not in only_ds and (exp_name, ds, s) not in full_train_map:
                    dup = dict(src_data)
                    dup["exp_name"] = exp_name
                    dup["exp_block"] = preset.get("block", "")
                    dup["exp_title"] = preset.get("title", "")
                    dup["exp_latex_name"] = preset.get("latex_name", exp_name)
                    full_train_map[(exp_name, ds, s)] = dup
            for (src_exp, ds, s), src_data in list(val_runs_map.items()):
                if src_exp == dup_from and ds not in only_ds and (exp_name, ds, s) not in val_runs_map:
                    dup = dict(src_data)
                    dup["exp_name"] = exp_name
                    dup["exp_block"] = preset.get("block", "")
                    dup["exp_title"] = preset.get("title", "")
                    dup["exp_latex_name"] = preset.get("latex_name", exp_name)
                    val_runs_map[(exp_name, ds, s)] = dup

    return full_train_map, val_runs_map


def _resolve_run_metrics(
    run_data: Dict[str, Any],
    val_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved = dict(run_data)
    if val_data is None:
        return resolved

    val_best_ep = val_data.get("best_epoch")
    if val_best_ep is None:
        return resolved

    val_best_ep = int(val_best_ep)
    resolved["val_selected_epoch"] = val_best_ep
    resolved["Val_RMSE"] = val_data.get("Val_RMSE")
    resolved["Val_S_Score"] = val_data.get("Val_S_Score")

    history = run_data.get("epoch_history", [])
    if isinstance(history, list) and history:
        for ep_item in history:
            if int(ep_item.get("epoch", -1)) == val_best_ep:
                resolved["best_epoch"] = val_best_ep
                resolved["Test_RMSE"] = ep_item["Test_RMSE"]
                resolved["Test_S_Score"] = ep_item["Test_S_Score"]
                for zk in ("RMSE_0_20", "RMSE_20_50", "RMSE_50_100", "RMSE_100_150"):
                    if zk in ep_item:
                        resolved[zk] = ep_item[zk]
                break

    return resolved


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    arr = np.array(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _fmt_md(mean: Optional[float], std: Optional[float], n: int, decimals: int = 2) -> str:
    if mean is None:
        return "—"
    if n <= 1:
        return f"{mean:.{decimals}f} ± 0.00"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _fmt_tex(mean: Optional[float], std: Optional[float], n: int, decimals: int = 2, bold: bool = False) -> str:
    if mean is None:
        return "--"
    val_str = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}" if n > 1 else f"{mean:.{decimals}f}"
    if bold:
        return f"$\\mathbf{{{val_str}}}$"
    return f"${val_str}$"


def generate_all_reports(
    out_dir: str,
    prev_results_dirs: Optional[List[str]] = None,
    expected_seeds: Optional[List[int]] = None,
    quiet: bool = False,
) -> Dict[str, str]:
    if expected_seeds is None:
        expected_seeds = DEFAULT_SEEDS

    os.makedirs(out_dir, exist_ok=True)
    full_train_raw, val_runs_map = discover_summaries_by_mode(out_dir, prev_results_dirs)

    summaries_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for key, r_data in full_train_raw.items():
        summaries_map[key] = _resolve_run_metrics(r_data, val_runs_map.get(key))

    using_val_fallback = False
    if not summaries_map and val_runs_map:
        summaries_map = dict(val_runs_map)
        using_val_fallback = True

    exp_order = list(ABLATION_GRID.keys())
    all_keys = set(summaries_map.keys()) | set(val_runs_map.keys())
    for exp_name, _, _ in sorted(all_keys):
        if exp_name not in exp_order:
            exp_order.append(exp_name)

    val_summary_md_path = os.path.join(out_dir, "val_epochs_summary.md")
    if val_runs_map:
        val_lines = [
            "# 🎯 Validation Best Epochs Summary (85% Train / 15% Val)",
            "",
            "| Experiment | Dataset | Seed | Selected Epoch (`best_epoch`) | Raw Min Epoch | Min `Val_RMSE` | Smooth-3 Center Epoch | Smooth-3 `Val_RMSE` | `Val_S_Score` | `Test_RMSE` (85% train) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        sorted_val = sorted(
            val_runs_map.values(),
            key=lambda r: (
                exp_order.index(r["exp_name"]) if r["exp_name"] in exp_order else 999,
                r["dataset"],
                int(r["seed"]),
            ),
        )
        for vr in sorted_val:
            raw_ep = vr.get("val_raw_best_epoch") or vr.get("best_epoch")
            smooth_ep = vr.get("val_smooth3_best_epoch")
            smooth_ep_str = f"Epoch {smooth_ep}" if smooth_ep is not None else "—"
            smooth_rmse_str = f"{vr['val_smooth3_rmse']:.2f}" if vr.get("val_smooth3_rmse") is not None else "—"
            v_rmse = f"{vr['Val_RMSE']:.2f}" if vr.get("Val_RMSE") is not None else "—"
            v_score = f"{vr['Val_S_Score']:.2f}" if vr.get("Val_S_Score") is not None else "—"
            t_rmse = f"{vr['Test_RMSE']:.2f}" if vr.get("Test_RMSE") is not None else "—"
            val_lines.append(
                f"| `{vr['exp_name']}` | {vr['dataset']} | `{vr['seed']}` | **Epoch {vr['best_epoch']}** | "
                f"Epoch {raw_ep} | {v_rmse} | {smooth_ep_str} | {smooth_rmse_str} | {v_score} | {t_rmse} |"
            )
        with open(val_summary_md_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(val_lines) + "\n")

    all_runs_csv_path = os.path.join(out_dir, "ablation_all_runs.csv")
    csv_cols = [
        "exp_name",
        "exp_block",
        "exp_title",
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
    sorted_runs = sorted(
        summaries_map.values(),
        key=lambda r: (
            exp_order.index(r["exp_name"]) if r["exp_name"] in exp_order else 999,
            r["dataset"],
            int(r["seed"]),
        ),
    )
    with open(all_runs_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted_runs:
            writer.writerow(row)

    total_target_runs = len(ABLATION_GRID) * len(DATASETS) * len(expected_seeds)
    completed_val_runs = sum(
        1
        for exp in ABLATION_GRID
        for ds in DATASETS
        for s in expected_seeds
        if (exp, ds, int(s)) in val_runs_map
    )
    completed_full_runs = sum(
        1
        for exp in ABLATION_GRID
        for ds in DATASETS
        for s in expected_seeds
        if (exp, ds, int(s)) in full_train_raw
    )
    pct_val = (100.0 * completed_val_runs / total_target_runs) if total_target_runs > 0 else 0.0
    pct_full = (100.0 * completed_full_runs / total_target_runs) if total_target_runs > 0 else 0.0

    all_recorded_runs = list(full_train_raw.values()) + list(val_runs_map.values())
    total_elapsed_sec = sum(float(r.get("elapsed_sec", 0.0) or 0.0) for r in all_recorded_runs)
    avg_sec_per_run = total_elapsed_sec / len(all_recorded_runs) if all_recorded_runs else 0.0

    progress_lines = [
        "# 📊 Ablation Study Progress Status (Multi-Seed)",
        "",
        f"- **Target Seeds ({len(expected_seeds)}):** `{expected_seeds}`",
        f"- **Stage 1 — Validation (85% Train / 15% Val):** **{completed_val_runs} / {total_target_runs}** ({pct_val:.1f}%)",
        f"- **Stage 2 — Full Train (100% Train -> Test):** **{completed_full_runs} / {total_target_runs}** ({pct_full:.1f}%)",
        f"- **Total GPU Time:** `{total_elapsed_sec / 3600.0:.2f} h` (avg `{avg_sec_per_run:.1f} sec/run`)",
        "",
        "## Completion Matrix (Full Train 100% | Validation 85/15 in parentheses)",
        "",
        "| Experiment | Block | FD001 | FD002 | FD003 | FD004 | 100% Train | 85/15 Val |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for exp_name in exp_order:
        preset = ALL_PRESETS.get(exp_name, {})
        block = preset.get("block", "Custom")
        ds_cells = []
        exp_full_done = 0
        exp_val_done = 0
        exp_total = len(DATASETS) * len(expected_seeds)

        for ds in DATASETS:
            k_full = sum(1 for s in expected_seeds if (exp_name, ds, int(s)) in full_train_raw)
            k_val = sum(1 for s in expected_seeds if (exp_name, ds, int(s)) in val_runs_map)
            exp_full_done += k_full
            exp_val_done += k_val
            icon = "✅" if k_full == len(expected_seeds) else ("⏳" if (k_full > 0 or k_val > 0) else "⬜")
            ds_cells.append(f"{icon} {k_full}/{len(expected_seeds)} (val:{k_val})")

        progress_lines.append(
            f"| `{exp_name}` | {block} | {ds_cells[0]} | {ds_cells[1]} | {ds_cells[2]} | {ds_cells[3]} | "
            f"**{exp_full_done}/{exp_total}** | **{exp_val_done}/{exp_total}** |"
        )

    progress_md_path = os.path.join(out_dir, "progress_status.md")
    with open(progress_md_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(progress_lines) + "\n")

    agg_stats: Dict[str, Dict[str, Any]] = {}
    for exp_name in exp_order:
        agg_stats[exp_name] = {"datasets": {}, "seed_counts": []}
        macro_rmse_means = []
        macro_score_means = []

        for ds in DATASETS:
            runs_ds = [
                r for (e, d, _), r in summaries_map.items() if e == exp_name and d == ds
            ]
            n_seeds = len(runs_ds)
            agg_stats[exp_name]["seed_counts"].append(n_seeds)

            rmses = [float(r["Test_RMSE"]) for r in runs_ds if r.get("Test_RMSE") is not None]
            scores = [float(r["Test_S_Score"]) for r in runs_ds if r.get("Test_S_Score") is not None]
            z0_20 = [float(r["RMSE_0_20"]) for r in runs_ds if r.get("RMSE_0_20") is not None]
            z20_50 = [float(r["RMSE_20_50"]) for r in runs_ds if r.get("RMSE_20_50") is not None]
            z50_100 = [float(r["RMSE_50_100"]) for r in runs_ds if r.get("RMSE_50_100") is not None]
            z100_150 = [float(r["RMSE_100_150"]) for r in runs_ds if r.get("RMSE_100_150") is not None]
            pcc_phys = [float(r["Median_PCC_Phys"]) for r in runs_ds if r.get("Median_PCC_Phys") is not None]
            pcc_rul = [float(r["Median_PCC_RUL"]) for r in runs_ds if r.get("Median_PCC_RUL") is not None]

            rmse_m, rmse_s = _mean_std(rmses)
            score_m, score_s = _mean_std(scores)

            if rmse_m is not None:
                macro_rmse_means.append(rmse_m)
            if score_m is not None:
                macro_score_means.append(score_m)

            agg_stats[exp_name]["datasets"][ds] = {
                "n": n_seeds,
                "rmse_mean": rmse_m,
                "rmse_std": rmse_s,
                "score_mean": score_m,
                "score_std": score_s,
                "z0_20": _mean_std(z0_20),
                "z20_50": _mean_std(z20_50),
                "z50_100": _mean_std(z50_100),
                "z100_150": _mean_std(z100_150),
                "pcc_phys": _mean_std(pcc_phys),
                "pcc_rul": _mean_std(pcc_rul),
            }

        agg_stats[exp_name]["macro_rmse"] = (
            float(np.mean(macro_rmse_means)) if macro_rmse_means else None
        )
        agg_stats[exp_name]["macro_score"] = (
            float(np.mean(macro_score_means)) if macro_score_means else None
        )

    sota_key = "00_full_model"
    sota_macro_rmse = agg_stats.get(sota_key, {}).get("macro_rmse", None)

    mode_note = (
        "> ⚠️ Note: Table is built from preliminary validation (85/15) runs since 100% full-train runs are not yet available."
        if using_val_fallback
        else "> ✅ Note: Where validation (85/15) runs exist, 100% train test metrics are evaluated at `val_best_epoch`."
    )
    summary_lines = [
        "# 📈 Ablation Study Summary (NASA C-MAPSS FD001–FD004)",
        "",
        mode_note,
        "",
        "## 1. Main Results: Test RMSE and NASA S-Score (Mean ± Std across seeds)",
        "",
        "| Experiment | Description | Seeds (N) | FD001 RMSE | FD001 S-Score | FD002 RMSE | FD002 S-Score | FD003 RMSE | FD003 S-Score | FD004 RMSE | FD004 S-Score | Macro RMSE | Macro S-Score | ΔRMSE vs 00_Full |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for exp_name in exp_order:
        preset = ALL_PRESETS.get(exp_name, {})
        title = preset.get("title", exp_name)
        st = agg_stats[exp_name]
        counts = st["seed_counts"]
        n_str = (
            str(counts[0])
            if len(set(counts)) == 1
            else "/".join(str(c) for c in counts)
        )

        ds_cols = []
        for ds in DATASETS:
            d_st = st["datasets"][ds]
            ds_cols.append(_fmt_md(d_st["rmse_mean"], d_st["rmse_std"], d_st["n"], decimals=2))
            ds_cols.append(_fmt_md(d_st["score_mean"], d_st["score_std"], d_st["n"], decimals=1))

        macro_rmse = st["macro_rmse"]
        macro_score = st["macro_score"]
        macro_rmse_str = f"**{macro_rmse:.2f}**" if macro_rmse is not None else "—"
        macro_score_str = f"{macro_score:.1f}" if macro_score is not None else "—"

        if exp_name == sota_key and macro_rmse is not None:
            delta_str = "0.00 (Ref)"
        elif macro_rmse is not None and sota_macro_rmse is not None:
            common_diffs = []
            for ds in DATASETS:
                m_exp = st["datasets"][ds]["rmse_mean"]
                m_sota = agg_stats[sota_key]["datasets"][ds]["rmse_mean"]
                if m_exp is not None and m_sota is not None:
                    common_diffs.append(m_exp - m_sota)
            if common_diffs:
                diff = float(np.mean(common_diffs))
                sign = "+" if diff >= 0 else ""
                delta_str = f"{sign}{diff:.2f}"
            else:
                delta_str = "—"
        else:
            delta_str = "—"

        summary_lines.append(
            f"| `{exp_name}` | {title} | {n_str} | "
            + " | ".join(ds_cols)
            + f" | {macro_rmse_str} | {macro_score_str} | {delta_str} |"
        )

    summary_lines.extend(
        [
            "",
            "## 2. Zone-wise RMSE Decomposition and Latent Isometry Metrics (SMART PCCs, Macro Avg)",
            "",
            "| Experiment | Critical `RMSE [0-20)` | Danger `RMSE [20-50)` | Transition `RMSE [50-100)` | Normal `RMSE [100-150)` | `Median_PCC_Phys` | `Median_PCC_RUL` |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for exp_name in exp_order:
        st = agg_stats[exp_name]
        z0_list, z20_list, z50_list, z100_list, pphys_list, prul_list = [], [], [], [], [], []
        for ds in DATASETS:
            d_st = st["datasets"][ds]
            if d_st["z0_20"][0] is not None:
                z0_list.append(d_st["z0_20"][0])
            if d_st["z20_50"][0] is not None:
                z20_list.append(d_st["z20_50"][0])
            if d_st["z50_100"][0] is not None:
                z50_list.append(d_st["z50_100"][0])
            if d_st["z100_150"][0] is not None:
                z100_list.append(d_st["z100_150"][0])
            if d_st["pcc_phys"][0] is not None:
                pphys_list.append(d_st["pcc_phys"][0])
            if d_st["pcc_rul"][0] is not None:
                prul_list.append(d_st["pcc_rul"][0])

        def _fmt_avg(lst: List[float], dec: int = 2) -> str:
            return f"{float(np.mean(lst)):.{dec}f}" if lst else "—"

        summary_lines.append(
            f"| `{exp_name}` | {_fmt_avg(z0_list)} | {_fmt_avg(z20_list)} | {_fmt_avg(z50_list)} | "
            f"{_fmt_avg(z100_list)} | {_fmt_avg(pphys_list, 3)} | {_fmt_avg(prul_list, 3)} |"
        )

    summary_md_path = os.path.join(out_dir, "ablation_summary.md")
    with open(summary_md_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    tex_lines = [
        "% Auto-generated by aggregate_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comprehensive Ablation Study on NASA C-MAPSS Benchmark Datasets (Mean $\\pm$ Std over 5 Random Seeds).}",
        "\\label{tab:ablation_study}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l c cc cc cc cc cc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{\\textbf{FD001}} & \\multicolumn{2}{c}{\\textbf{FD002}} & \\multicolumn{2}{c}{\\textbf{FD003}} & \\multicolumn{2}{c}{\\textbf{FD004}} & \\multicolumn{2}{c}{\\textbf{Macro Average}} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10} \\cmidrule(lr){11-12}",
        "\\textbf{Model / Ablation Variant} & \\textbf{Seeds} & \\textbf{RMSE} $\\downarrow$ & \\textbf{S-Score} $\\downarrow$ & \\textbf{RMSE} $\\downarrow$ & \\textbf{S-Score} $\\downarrow$ & \\textbf{RMSE} $\\downarrow$ & \\textbf{S-Score} $\\downarrow$ & \\textbf{RMSE} $\\downarrow$ & \\textbf{S-Score} $\\downarrow$ & \\textbf{RMSE} $\\downarrow$ & $\\Delta\\textbf{RMSE}$ \\\\",
        "\\midrule",
    ]

    last_block = None
    for exp_name in exp_order:
        preset = ALL_PRESETS.get(exp_name, {})
        block = preset.get("block", "Other")
        latex_label = preset.get("latex_name", exp_name.replace("_", "\\_"))
        is_sota = exp_name == sota_key

        if last_block is not None and block != last_block:
            tex_lines.append("\\midrule")
        last_block = block

        st = agg_stats[exp_name]
        counts = st["seed_counts"]
        n_str = str(counts[0]) if len(set(counts)) == 1 else "/".join(str(c) for c in counts)

        cells = []
        for ds in DATASETS:
            d_st = st["datasets"][ds]
            cells.append(_fmt_tex(d_st["rmse_mean"], d_st["rmse_std"], d_st["n"], decimals=2, bold=is_sota))
            cells.append(_fmt_tex(d_st["score_mean"], d_st["score_std"], d_st["n"], decimals=1, bold=is_sota))

        macro_rmse = st["macro_rmse"]
        if macro_rmse is not None:
            macro_str = f"$\\mathbf{{{macro_rmse:.2f}}}$" if is_sota else f"${macro_rmse:.2f}$"
        else:
            macro_str = "--"

        if is_sota and macro_rmse is not None:
            delta_tex = "\\textbf{0.00}"
        elif macro_rmse is not None and sota_macro_rmse is not None:
            common_diffs = []
            for ds in DATASETS:
                m_exp = st["datasets"][ds]["rmse_mean"]
                m_sota = agg_stats[sota_key]["datasets"][ds]["rmse_mean"]
                if m_exp is not None and m_sota is not None:
                    common_diffs.append(m_exp - m_sota)
            if common_diffs:
                diff = float(np.mean(common_diffs))
                sign = "+" if diff >= 0 else ""
                delta_tex = f"${sign}{diff:.2f}$"
            else:
                delta_tex = "--"
        else:
            delta_tex = "--"

        row_str = f"{latex_label} & {n_str} & " + " & ".join(cells) + f" & {macro_str} & {delta_tex} \\\\"
        tex_lines.append(row_str)

    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )

    tex_path = os.path.join(out_dir, "ablation_table.tex")
    with open(tex_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    if not quiet:
        print("\n" + "=" * 80)
        print(f"AGGREGATION REPORTS SAVED TO: {os.path.abspath(out_dir)}")
        print(
            f"   - Validation Runs (85/15): {completed_val_runs}/{total_target_runs} ({pct_val:.1f}%)"
        )
        print(
            f"   - Full Train Runs (100%):  {completed_full_runs}/{total_target_runs} ({pct_full:.1f}%)"
        )
        print(f"   - Progress Status:         {progress_md_path}")
        if val_runs_map:
            print(f"   - Best Val Epochs:         {val_summary_md_path}")
        print(f"   - All Runs (CSV):          {all_runs_csv_path}")
        print(f"   - Summary (Markdown):      {summary_md_path}")
        print(f"   - Summary (LaTeX):         {tex_path}")
        print("=" * 80)

    return {
        "progress_status_md": progress_md_path,
        "val_epochs_summary_md": val_summary_md_path,
        "ablation_all_runs_csv": all_runs_csv_path,
        "ablation_summary_md": summary_md_path,
        "ablation_table_tex": tex_path,
    }


def parse_args() -> argparse.Namespace:
    default_out = "/kaggle/working/results" if os.path.exists("/kaggle") else "results"
    parser = argparse.ArgumentParser(
        description="Aggregate partial or complete Ablation Study results (Mean ± Std across seeds) into Markdown, CSV, and LaTeX."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out,
        help="Working directory with results (default: /kaggle/working/results or ./results).",
    )
    parser.add_argument(
        "--prev_results_dir",
        type=str,
        nargs="*",
        default=None,
        help="Optional path(s) to previous session results.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Expected seeds for N/5 progress calculation (default: 42 43 44 45 50).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_reports(
        out_dir=args.out_dir,
        prev_results_dirs=args.prev_results_dir,
        expected_seeds=args.seeds,
        quiet=False,
    )
