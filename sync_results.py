import argparse
import os
import shutil
import zipfile
from typing import List, Optional, Tuple

from aggregate_results import generate_all_reports
from src.trainer import append_to_master_ledger, get_run_dir, is_valid_summary_file


def sync_and_package_results(
    out_dir: str,
    prev_results_dirs: Optional[List[str]] = None,
    zip_path: Optional[str] = None,
    overwrite: bool = False,
    include_weights: bool = False,
) -> Tuple[int, int, str]:
    """
    1. Копирует все ранее посчитанные прогоны (metrics_summary.json, epoch_log.csv, графики)
       из read-only папок prev_results_dirs (/kaggle/input/...) в рабочую папку out_dir (/kaggle/working/results).
    2. Пересобирает master_ledger.jsonl / master_ledger.csv и сводные таблицы (Markdown, CSV, LaTeX).
    3. Упаковывает объединенную папку out_dir в единый компактный архив results_backup.zip.
    """
    os.makedirs(out_dir, exist_ok=True)
    copied_runs = 0
    skipped_runs = 0

    if prev_results_dirs:
        for prev_dir in prev_results_dirs:
            if not prev_dir or not os.path.isdir(prev_dir):
                print(f"[WARN] Папка прошлых результатов не найдена или недоступна: {prev_dir}")
                continue

            print(f"[SYNC] Сканируем прошлые результаты в: {prev_dir} ...")
            for root, _, files in os.walk(prev_dir):
                if "metrics_summary.json" not in files:
                    continue

                src_summary_path = os.path.join(root, "metrics_summary.json")
                summary = is_valid_summary_file(src_summary_path)
                if summary is None:
                    continue

                exp_name = str(summary["exp_name"])
                dataset_name = str(summary["dataset"]).upper()
                seed = int(summary["seed"])
                is_val = bool(summary.get("use_validation", False)) or ("runs_val" in root.replace("\\", "/"))

                dst_run_dir = get_run_dir(out_dir, exp_name, dataset_name, seed, use_validation=is_val)
                dst_summary_path = os.path.join(dst_run_dir, "metrics_summary.json")

                if not overwrite and is_valid_summary_file(dst_summary_path) is not None:
                    skipped_runs += 1
                    continue

                os.makedirs(dst_run_dir, exist_ok=True)
                for item in os.listdir(root):
                    if not include_weights and item.endswith(".pth"):
                        continue
                    src_item = os.path.join(root, item)
                    dst_item = os.path.join(dst_run_dir, item)
                    if os.path.abspath(src_item) != os.path.abspath(dst_item):
                        if os.path.isdir(src_item):
                            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                        elif os.path.isfile(src_item):
                            shutil.copy2(src_item, dst_item)

                copied_runs += 1

    # Синхронизируем master_ledger по всем прогонам в out_dir
    total_merged = 0
    for root, _, files in os.walk(out_dir):
        if "metrics_summary.json" in files:
            summary = is_valid_summary_file(os.path.join(root, "metrics_summary.json"))
            if summary is not None:
                append_to_master_ledger(out_dir, summary)
                total_merged += 1

    # Обновляем все сводные таблицы внутри out_dir перед архивацией
    generate_all_reports(out_dir=out_dir, prev_results_dirs=None, quiet=True)

    # Упаковываем всё в компактный архив results_backup.zip
    if zip_path is None:
        parent_dir = os.path.dirname(os.path.abspath(out_dir))
        zip_path = os.path.join(parent_dir, "results_backup.zip")

    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    abs_zip_path = os.path.abspath(zip_path)
    abs_out_dir = os.path.abspath(out_dir)

    archived_files = 0
    with zipfile.ZipFile(abs_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(abs_out_dir):
            for fname in files:
                if not include_weights and fname.endswith(".pth"):
                    continue
                if ".tmp." in fname:
                    continue
                full_path = os.path.join(root, fname)
                if os.path.abspath(full_path) == abs_zip_path:
                    continue
                arcname = os.path.relpath(full_path, start=abs_out_dir)
                zf.write(full_path, arcname=arcname)
                archived_files += 1

    zip_size_mb = os.path.getsize(abs_zip_path) / (1024 * 1024)
    print("\n" + "=" * 75)
    print("SYNC AND ARCHIVE COMPLETED:")
    print(f"   - Copied from previous sessions: {copied_runs} runs (skipped duplicates: {skipped_runs})")
    print(f"   - Total merged runs in output dir: {total_merged}")
    print(f"   - Archive ready: {abs_zip_path} ({zip_size_mb:.2f} MB, files: {archived_files})")
    print("=" * 75)

    return copied_runs, total_merged, abs_zip_path


def parse_args() -> argparse.Namespace:
    is_kaggle = os.path.exists("/kaggle")
    default_out = "/kaggle/working/results" if is_kaggle else "results"
    default_zip = "/kaggle/working/results_backup.zip" if is_kaggle else "results_backup.zip"

    parser = argparse.ArgumentParser(
        description="Слияние результатов прошлых сессий Kaggle с текущими и упаковка в results_backup.zip."
    )
    parser.add_argument(
        "--prev_results_dir",
        type=str,
        nargs="*",
        default=None,
        help="Путь (или список путей) к подключенному Kaggle Dataset с прошлыми результатами (/kaggle/input/...).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out,
        help="Рабочая папка результатов текущей сессии (по умолчанию /kaggle/working/results или ./results).",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=default_zip,
        help="Путь для сохранения итогового zip-архива (по умолчанию /kaggle/working/results_backup.zip).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать существующие прогоны в out_dir файлами из prev_results_dir.",
    )
    parser.add_argument(
        "--include_weights",
        action="store_true",
        help="Включать ли тяжелые .pth чекпоинты при копировании и упаковке в zip (по умолчанию False).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync_and_package_results(
        out_dir=args.out_dir,
        prev_results_dirs=args.prev_results_dir,
        zip_path=args.zip_path,
        overwrite=args.overwrite,
        include_weights=args.include_weights,
    )
