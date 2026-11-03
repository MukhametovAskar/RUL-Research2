import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from configs.base_config import COLS, DROP_SENSORS


def resolve_cmapss_data_dir(
    preferred_dir: str = "/kaggle/input/datasets/kylibiaka/cmapss",
    dataset_name: str = "FD001",
) -> str:
    target_file = f"train_{dataset_name.upper()}.txt"

    if preferred_dir and os.path.isfile(os.path.join(preferred_dir, target_file)):
        return preferred_dir

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_dirs = [
        os.path.join(repo_root, "data"),
        os.path.abspath("data"),
        "/kaggle/input/datasets/kylibiaka/cmapss",
        "/kaggle/input/datasets/mukhametovaskar/cmapss",
        "/kaggle/input/cmapss",
        "/kaggle/input/nasa-cmaps",
        "/kaggle/input/nasa-cmapss",
    ]
    for cand in candidate_dirs:
        if os.path.isfile(os.path.join(cand, target_file)):
            return cand

    search_roots = []
    if preferred_dir and os.path.isdir(preferred_dir):
        search_roots.append(preferred_dir)
    if os.path.isdir("/kaggle/input"):
        search_roots.append("/kaggle/input")

    for s_root in search_roots:
        for root, _, files in os.walk(s_root):
            if target_file in files:
                return root

    raise FileNotFoundError(
        f"Could not find '{target_file}' in '{preferred_dir}', './data', or '/kaggle/input'. "
        f"Please specify a valid path via --data_dir or data_root=..."
    )


def apply_regime_specific_normalization(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_clusters: int = 6,
    df_val: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    op_cols = ["op1", "op2", "op3"]
    sensor_cols = [c for c in df_train.columns if c.startswith("s")]

    df_train[sensor_cols] = df_train[sensor_cols].astype(np.float32)
    df_test[sensor_cols] = df_test[sensor_cols].astype(np.float32)
    if df_val is not None:
        df_val[sensor_cols] = df_val[sensor_cols].astype(np.float32)

    # KMeans и экстремумы считаются СТРОГО только по df_train (без утечки из валидации или теста)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_train[op_cols])

    df_train["regime_idx"] = kmeans.predict(df_train[op_cols])
    df_test["regime_idx"] = kmeans.predict(df_test[op_cols])
    if df_val is not None:
        df_val["regime_idx"] = kmeans.predict(df_val[op_cols])

    for cluster_id in range(n_clusters):
        tr_mask = df_train["regime_idx"] == cluster_id
        te_mask = df_test["regime_idx"] == cluster_id
        if tr_mask.sum() == 0:
            continue

        mins = df_train.loc[tr_mask, sensor_cols].min()
        maxs = df_train.loc[tr_mask, sensor_cols].max()
        diffs = maxs - mins
        diffs[diffs == 0] = 1.0

        df_train.loc[tr_mask, sensor_cols] = (df_train.loc[tr_mask, sensor_cols] - mins) / diffs
        if te_mask.sum() > 0:
            df_test.loc[te_mask, sensor_cols] = (df_test.loc[te_mask, sensor_cols] - mins) / diffs
        if df_val is not None:
            val_mask = df_val["regime_idx"] == cluster_id
            if val_mask.sum() > 0:
                df_val.loc[val_mask, sensor_cols] = (df_val.loc[val_mask, sensor_cols] - mins) / diffs

    return df_train, df_test, df_val


def load_raw_data(
    train_path: str,
    test_path: str,
    rul_path: str,
    max_rul: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    train_df.drop(columns=DROP_SENSORS, inplace=True, errors="ignore")
    test_df.drop(columns=DROP_SENSORS, inplace=True, errors="ignore")

    features = [c for c in train_df.columns if c.startswith("s")]

    rul_train = pd.DataFrame(train_df.groupby("unit")["time"].max()).reset_index()
    rul_train.columns = ["unit", "max_time"]
    train_df = train_df.merge(rul_train, on="unit", how="left")
    train_df["RUL"] = train_df["max_time"] - train_df["time"]
    train_df.drop(columns=["max_time"], inplace=True)
    train_df["RUL"] = train_df["RUL"].clip(upper=max_rul)

    rul_test = pd.DataFrame(test_df.groupby("unit")["time"].max()).reset_index()
    rul_test.columns = ["unit", "max_time"]
    true_rul["unit"] = true_rul.index + 1
    true_rul["max_rul"] = true_rul["RUL"]
    rul_test = rul_test.merge(true_rul[["unit", "max_rul"]], on="unit", how="left")
    rul_test["total_life"] = rul_test["max_time"] + rul_test["max_rul"]

    test_df = test_df.merge(rul_test[["unit", "total_life"]], on="unit", how="left")
    test_df["RUL"] = test_df["total_life"] - test_df["time"]
    test_df.drop(columns=["total_life"], inplace=True)
    test_df["RUL"] = test_df["RUL"].clip(upper=max_rul)

    return train_df, test_df, features


def extract_windows(
    df: pd.DataFrame, feature_cols: List[str], seq_len: int, is_test: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, units = [], [], []
    for unit_id in df["unit"].unique():
        unit_data = df[df["unit"] == unit_id].sort_values("time")
        data_matrix = unit_data[feature_cols].values
        rul_array = unit_data["RUL"].values
        L = len(data_matrix)

        if is_test:
            if L >= seq_len:
                x.append(data_matrix[-seq_len:])
            else:
                pad_len = seq_len - L
                padded_matrix = np.pad(data_matrix, ((pad_len, 0), (0, 0)), mode="edge")
                x.append(padded_matrix)
            y.append(rul_array[-1])
            units.append(unit_id)
        else:
            if L >= seq_len:
                for i in range(L - seq_len + 1):
                    x.append(data_matrix[i : i + seq_len])
                    y.append(rul_array[i + seq_len - 1])
                    units.append(unit_id)
            else:
                pad_len = seq_len - L
                padded_matrix = np.pad(data_matrix, ((pad_len, 0), (0, 0)), mode="edge")
                x.append(padded_matrix)
                y.append(rul_array[-1])
                units.append(unit_id)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(units)


class CMAPSSDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def prepare_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, Optional[DataLoader], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset_name = cfg["dataset_name"]
    data_dir = resolve_cmapss_data_dir(cfg.get("data_dir", ""), dataset_name=dataset_name)

    train_path = os.path.join(data_dir, f"train_{dataset_name}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset_name}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset_name}.txt")

    train_df, test_df, features = load_raw_data(
        train_path=train_path,
        test_path=test_path,
        rul_path=rul_path,
        max_rul=cfg["max_rul"],
    )

    norm_mode = cfg.get("norm_mode", "regime")
    n_clusters = 6 if (norm_mode == "regime" and dataset_name in ["FD002", "FD004"]) else 1
    cols_to_drop = ["op1", "op2", "op3", "regime_idx"]

    if cfg.get("use_validation", False):
        # Сплит двигателей на трейн и валидацию выполняется ДО нормализации и KMeans (без утечки данных)
        all_units = train_df["unit"].unique()
        num_val_units = int(len(all_units) * cfg.get("val_split", 0.15))

        val_units = np.random.choice(all_units, num_val_units, replace=False)
        train_units = np.setdiff1d(all_units, val_units)

        print(
            f"[{dataset_name}] Validation mode: Train units = {len(train_units)} | Val units = {len(val_units)}"
        )

        real_train_df = train_df[train_df["unit"].isin(train_units)].copy()
        val_df = train_df[train_df["unit"].isin(val_units)].copy()

        if n_clusters == 6:
            print(f"[{dataset_name}] Applying Regime-Specific Normalization (6 clusters, fit on train split only)...")
        else:
            print(f"[{dataset_name}] Applying Global Normalization (1 cluster, fit on train split only)...")

        real_train_df, test_df, val_df = apply_regime_specific_normalization(
            real_train_df, test_df, n_clusters=n_clusters, df_val=val_df
        )

        real_train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        val_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        test_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        X_train, y_train, _ = extract_windows(real_train_df, features, cfg["seq_len"], is_test=False)
        X_val, y_val, _ = extract_windows(val_df, features, cfg["seq_len"], is_test=False)
        val_loader = DataLoader(CMAPSSDataset(X_val, y_val), batch_size=cfg["batch_size"], shuffle=False)
    else:
        print(f"[{dataset_name}] Full train mode (100% train data).")
        if n_clusters == 6:
            print(f"[{dataset_name}] Applying Regime-Specific Normalization (6 clusters)...")
        else:
            print(f"[{dataset_name}] Applying Global Normalization (1 cluster)...")

        train_df, test_df, _ = apply_regime_specific_normalization(
            train_df, test_df, n_clusters=n_clusters, df_val=None
        )

        train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        test_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        X_train, y_train, _ = extract_windows(train_df, features, cfg["seq_len"], is_test=False)
        val_loader = None

    X_test, y_test, _ = extract_windows(test_df, features, cfg["seq_len"], is_test=True)
    train_loader = DataLoader(
        CMAPSSDataset(X_train, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, X_train, y_train, X_test, y_test, features
