import argparse
import os
import sys
from typing import List, Optional

from aggregate_results import generate_all_reports
from configs.ablation_grid import (
    ABLATION_GRID,
    ALL_PRESETS,
    get_ablation_config,
    resolve_experiment_names,
)
from configs.base_config import DATASETS, DEFAULT_SEEDS
from src.trainer import (
    append_to_master_ledger,
    duplicate_run_summary,
    find_existing_summary,
    train_and_evaluate_single_run,
)


def resolve_datasets(ds_args: List[str]) -> List[str]:
    if not ds_args or any(d.lower() == "all" for d in ds_args):
        return list(DATASETS)
    resolved: List[str] = []
    for d in ds_args:
        du = d.strip().upper()
        if du not in DATASETS:
            raise ValueError(f"Unknown dataset '{d}'. Valid datasets: {DATASETS}")
        if du not in resolved:
            resolved.append(du)
    return resolved


def resolve_seeds(seed_args: List[str]) -> List[int]:
    if not seed_args:
        return [50]
    if any(str(s).lower() == "all" for s in seed_args):
        return list(DEFAULT_SEEDS)
    resolved: List[int] = []
    for s in seed_args:
        val = int(s)
        if val not in resolved:
            resolved.append(val)
    return resolved


def print_ablation_registry() -> None:
    print("\n" + "=" * 90)
    print("ABLATION STUDY EXPERIMENTS REGISTRY (00 - 07):")
    print("=" * 90)
    for name, info in ABLATION_GRID.items():
        print(f"  [{name}] ({info['block']})")
        print(f"      Title: {info['title']}")
        print(f"      Desc : {info['description']}")
        if info["overrides"]:
            print(f"      Flags: {info['overrides']}")
    print("=" * 90 + "\n")


def parse_args() -> argparse.Namespace:
    is_kaggle = os.path.exists("/kaggle")
    default_out = "/kaggle/working/results" if is_kaggle else "results"

    parser = argparse.ArgumentParser(
        description="CLI runner for Ablation Study on NASA C-MAPSS (FD001-FD004) with automatic resume."
    )
    parser.add_argument(
        "--exps",
        "--exp",
        dest="exps",
        type=str,
        nargs="+",
        default=["all"],
        help="Experiment names or prefixes (e.g. --exps 00 01, --exps 0 1 2, or --exps all).",
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        dest="datasets",
        type=str,
        nargs="+",
        default=["all"],
        help="Datasets to run (e.g. --datasets FD001 FD003 or --datasets all).",
    )
    parser.add_argument(
        "--seeds",
        "--seed",
        dest="seeds",
        type=str,
        nargs="+",
        default=["50"],
        help="One or more seeds (e.g. --seed 50, --seeds 42 43 44 45 50, or --seeds all).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/datasets/kylibiaka/cmapss",
        help="Path to C-MAPSS dataset directory (auto-resolves ./data and /kaggle/input).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out,
        help="Output directory for results (default: /kaggle/working/results on Kaggle or ./results locally).",
    )
    parser.add_argument(
        "--prev_results_dir",
        type=str,
        nargs="*",
        default=None,
        help="Optional path(s) to previous session results for incremental resume.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip already completed runs found in --out_dir or --prev_results_dir (enabled by default).",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Disable automatic skipping of completed runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force recomputation even if metrics_summary.json exists.",
    )
    parser.add_argument(
        "--save_weights",
        action="store_true",
        default=False,
        help="Save .pth model checkpoints for all runs.",
    )
    parser.add_argument(
        "--save_weights_seed50",
        action="store_true",
        default=False,
        help="Save .pth model checkpoints only for the default seed=50.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Render diagnostic and result plots for all seeds (enabled by default).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="Disable plot generation completely.",
    )
    parser.add_argument(
        "--no_tb",
        action="store_true",
        default=False,
        help="Disable TensorBoard event logging.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional override for learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for batch size.",
    )
    parser.add_argument(
        "--num_proxies",
        type=int,
        default=None,
        help="Optional override for number of proxies K.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Optional override for temperature.",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=None,
        help="Optional override for KL loss weight.",
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=None,
        help="Optional override for MSE loss weight.",
    )
    parser.add_argument(
        "--lambda_order",
        type=float,
        default=None,
        help="Optional override for Cosine Gram order regularizer weight.",
    )
    parser.add_argument(
        "--use_validation",
        action="store_true",
        default=False,
        help="Run validation mode (85% train / 15% val split before normalization).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of engines for validation when --use_validation is set (default: 0.15).",
    )
    parser.add_argument(
        "--target_epoch",
        type=int,
        default=None,
        help="Explicitly specify target epoch when training on 100% data.",
    )
    parser.add_argument(
        "--val_select_mode",
        type=str,
        choices=["min", "smooth3", "auto"],
        default="min",
        help="Validation epoch selection strategy: 'min' (pointwise minimum Val_RMSE), 'smooth3' (center of 3-epoch minimum window), or 'auto' (smooth3 if raw min occurs >2 epochs earlier than smooth3).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all available ablation experiments (00 - 07) and exit.",
    )
    return parser.parse_args()


def _check_existing_across_dirs(
    exp_name: str,
    dataset_name: str,
    seed: int,
    out_dir: str,
    prev_results_dirs: Optional[List[str]],
    use_validation: bool = False,
):
    existing = find_existing_summary(
        exp_name, dataset_name, seed, out_dir=out_dir, prev_results_dir=None, use_validation=use_validation
    )
    if existing is not None:
        return existing
    if prev_results_dirs:
        for p_dir in prev_results_dirs:
            existing = find_existing_summary(
                exp_name, dataset_name, seed, out_dir=out_dir, prev_results_dir=p_dir, use_validation=use_validation
            )
            if existing is not None:
                return existing
    return None


def main() -> None:
    args = parse_args()

    if args.list:
        print_ablation_registry()
        sys.exit(0)

    exp_names = resolve_experiment_names(args.exps)
    datasets = resolve_datasets(args.datasets)
    seeds = resolve_seeds(args.seeds)

    os.makedirs(args.out_dir, exist_ok=True)

    total_planned = len(exp_names) * len(datasets) * len(seeds)
    mode_desc = (
        "VALIDATION (85% Train / 15% Val -> find best epoch)"
        if args.use_validation
        else "FULL TRAIN (100% Train -> Test)"
    )
    print("\n" + "=" * 80)
    print("STARTING ABLATION STUDY MATRIX (STAR-Proxy RUL)")
    print(f"   - Mode:                    {mode_desc}")
    print(f"   - Experiments ({len(exp_names)}): {exp_names}")
    print(f"   - Datasets ({len(datasets)}):    {datasets}")
    print(f"   - Seeds ({len(seeds)}):       {seeds}")
    print(f"   - Total queued runs:       {total_planned}")
    print(f"   - Output dir (--out_dir):  {os.path.abspath(args.out_dir)}")
    if args.prev_results_dir:
        print(f"   - Previous results dir:    {args.prev_results_dir}")
    print("=" * 80)

    completed_count = 0
    skipped_count = 0

    for exp_name in exp_names:
        preset = ALL_PRESETS.get(exp_name, {})
        only_datasets = preset.get("only_datasets", None)
        duplicate_from = preset.get("duplicate_from", None)

        for dataset_name in datasets:
            for seed in seeds:
                if args.resume and not args.force:
                    existing_summary = _check_existing_across_dirs(
                        exp_name=exp_name,
                        dataset_name=dataset_name,
                        seed=seed,
                        out_dir=args.out_dir,
                        prev_results_dirs=args.prev_results_dir,
                        use_validation=args.use_validation,
                    )
                    if existing_summary is not None:
                        metric_val = (
                            f"Val_RMSE={existing_summary.get('Val_RMSE')}"
                            if args.use_validation
                            else f"Test_RMSE={existing_summary.get('Test_RMSE')}"
                        )
                        print(
                            f"[SKIP] Already completed: {exp_name} | {dataset_name} | seed={seed} "
                            f"(Best_Epoch={existing_summary.get('best_epoch')}, {metric_val})"
                        )
                        append_to_master_ledger(args.out_dir, existing_summary)
                        skipped_count += 1
                        continue

                # Для 07_wo_regime_normalization на однорежимных FD001 и FD003 дублируем результат из 00_full_model
                if only_datasets is not None and dataset_name not in only_datasets and duplicate_from:
                    ref_summary = _check_existing_across_dirs(
                        exp_name=duplicate_from,
                        dataset_name=dataset_name,
                        seed=seed,
                        out_dir=args.out_dir,
                        prev_results_dirs=args.prev_results_dir,
                        use_validation=args.use_validation,
                    )
                    if ref_summary is not None:
                        dup_summary = duplicate_run_summary(
                            source_summary=ref_summary,
                            target_exp_name=exp_name,
                            target_preset=preset,
                            out_dir=args.out_dir,
                            prev_results_dirs=args.prev_results_dir,
                        )
                        print(
                            f"[DUPLICATE] {exp_name} | {dataset_name} | seed={seed} <- copied from {duplicate_from} "
                            f"(single operating regime, Test_RMSE={dup_summary.get('Test_RMSE')})"
                        )
                        completed_count += 1
                        continue

                cfg = get_ablation_config(exp_name=exp_name, dataset_name=dataset_name, seed=seed)
                cfg["data_dir"] = args.data_dir
                cfg["save_dir"] = args.out_dir
                cfg["use_validation"] = args.use_validation
                cfg["val_split"] = args.val_split
                cfg["val_select_mode"] = args.val_select_mode
                if args.epochs is not None:
                    cfg["epochs"] = int(args.epochs)
                if args.lr is not None:
                    cfg["lr"] = float(args.lr)
                if args.batch_size is not None:
                    cfg["batch_size"] = int(args.batch_size)
                if args.num_proxies is not None:
                    cfg["num_proxies"] = int(args.num_proxies)
                if args.temp is not None:
                    cfg["temp"] = float(args.temp)
                if args.kl_weight is not None:
                    cfg["kl_weight"] = float(args.kl_weight)
                if args.mse_weight is not None:
                    cfg["mse_weight"] = float(args.mse_weight)
                if args.lambda_order is not None:
                    cfg["lambda_order"] = float(args.lambda_order)

                should_save_weights = args.save_weights or (args.save_weights_seed50 and seed == 50)
                should_save_plots = False if args.no_plots else bool(args.save_plots)

                train_and_evaluate_single_run(
                    cfg=cfg,
                    out_dir=args.out_dir,
                    save_weights=should_save_weights,
                    save_plots=should_save_plots,
                    prev_results_dirs=args.prev_results_dir,
                    target_epoch=args.target_epoch,
                    enable_tb=not args.no_tb,
                )
                completed_count += 1

    print("\n" + "=" * 80)
    print(
        f"QUEUE COMPLETED: newly trained = {completed_count} | skipped (Resume) = {skipped_count} | total = {total_planned}"
    )
    print("=" * 80)

    generate_all_reports(
        out_dir=args.out_dir,
        prev_results_dirs=args.prev_results_dir,
        expected_seeds=DEFAULT_SEEDS,
        quiet=False,
    )


if __name__ == "__main__":
    main()
