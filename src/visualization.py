import os
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

try:
    import seaborn as sns
except ImportError:
    sns = None


def calc_row_pcc(D1: torch.Tensor, D2: torch.Tensor) -> np.ndarray:
    D1_c = D1 - D1.mean(dim=1, keepdim=True)
    D2_c = D2 - D2.mean(dim=1, keepdim=True)
    num = (D1_c * D2_c).sum(dim=1)
    den = torch.sqrt((D1_c ** 2).sum(dim=1) * (D2_c ** 2).sum(dim=1) + 1e-9)
    return (num / den).cpu().numpy()


def compute_zone_rmses(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    zones = [
        ("RMSE_0_20", 0, 20),
        ("RMSE_20_50", 20, 50),
        ("RMSE_50_100", 50, 100),
        ("RMSE_100_150", 100, 150),
    ]
    zone_metrics: Dict[str, float] = {}
    for key, low, high in zones:
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        else:
            rmse = 0.0
        zone_metrics[key] = round(rmse, 4)
    return zone_metrics


def compute_smart_pccs(
    encoder: torch.nn.Module,
    data_loader: DataLoader,
    cfg: dict,
    max_samples: int = 1000,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    was_training = encoder.training
    encoder.eval()

    all_embs = []
    all_phys_feats = []
    all_ruls = []

    P = cfg["patch_size"]
    device = cfg["device"]

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = encoder(batch_x)
            emb_norm = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else None
            if emb_norm is None:
                continue

            q = max(P, batch_x.size(1) // 4)
            x_end = batch_x[:, -q:, :].mean(dim=1)
            x_start = batch_x[:, :q, :].mean(dim=1)
            phys_feat = torch.cat([x_end, x_end - x_start], dim=-1)

            all_embs.append(emb_norm.cpu())
            all_phys_feats.append(phys_feat.cpu())
            all_ruls.append(batch_y.cpu())

    if was_training:
        encoder.train()

    if not all_embs:
        empty = np.zeros(1, dtype=np.float32)
        return 0.0, 0.0, empty, empty

    embs_cat = torch.cat(all_embs, dim=0)
    phys_cat = torch.cat(all_phys_feats, dim=0)
    ruls_cat = torch.cat(all_ruls, dim=0)

    m_sub = min(max_samples, len(embs_cat))
    gen = torch.Generator()
    gen.manual_seed(42)
    idx_sub = torch.randperm(len(embs_cat), generator=gen)[:m_sub]

    z_sub = embs_cat[idx_sub]
    x_sub = phys_cat[idx_sub]
    y_sub = ruls_cat[idx_sub].unsqueeze(1)

    D_lat = torch.cdist(z_sub, z_sub, p=2)
    D_phys = torch.cdist(x_sub, x_sub, p=2)
    D_rul = torch.cdist(y_sub, y_sub, p=1)

    pcc_phys = calc_row_pcc(D_phys, D_lat)
    pcc_rul = calc_row_pcc(D_rul, D_lat)

    med_phys = float(np.median(pcc_phys))
    med_rul = float(np.median(pcc_rul))
    return round(med_phys, 4), round(med_rul, 4), pcc_phys, pcc_rul


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_emb: Optional[np.ndarray],
    y_train_subset: Optional[np.ndarray],
    proxies_emb: Optional[np.ndarray],
    proxy_ruls: Optional[np.ndarray],
    dataset_name: str,
    save_dir: str,
    show: bool = False,
) -> plt.Figure:
    if sns is not None:
        sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"Proxy-Anchor Metric Learning: {dataset_name}", fontsize=18, fontweight="bold", y=1.05)

    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor="k", c="dodgerblue", s=30)
    axes[0].plot([0, 130], [0, 130], "r--", linewidth=2, label="Идеал (True = Pred)")
    axes[0].set_xlabel("Истинный RUL (True)", fontsize=12)
    axes[0].set_ylabel("Предсказанный RUL (Pred)", fontsize=12)
    axes[0].set_title("True vs Predicted RUL (Test Set)", fontsize=14, fontweight="bold")
    axes[0].legend()

    if train_emb is not None and y_train_subset is not None and len(train_emb) > 10:
        print(f"[{dataset_name}] Computing t-SNE latent space...")
        tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
        if proxies_emb is not None and proxy_ruls is not None and len(proxy_ruls) > 0:
            combined_emb = np.vstack([train_emb, proxies_emb])
            combined_2d = tsne.fit_transform(combined_emb)
            num_proxies = len(proxy_ruls)
            emb_2d = combined_2d[:-num_proxies]
            proxies_2d = combined_2d[-num_proxies:]

            sc = axes[1].scatter(
                emb_2d[:, 0], emb_2d[:, 1], c=y_train_subset, cmap="coolwarm_r", s=15, alpha=0.5
            )
            axes[1].scatter(
                proxies_2d[:, 0],
                proxies_2d[:, 1],
                c=proxy_ruls,
                cmap="coolwarm_r",
                marker="*",
                s=400,
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
            )
            axes[1].set_title(
                "t-SNE: Данные (точки) и Аттракторы (звезды)", fontsize=14, fontweight="bold"
            )
        else:
            emb_2d = tsne.fit_transform(train_emb)
            sc = axes[1].scatter(
                emb_2d[:, 0], emb_2d[:, 1], c=y_train_subset, cmap="coolwarm_r", s=15, alpha=0.5
            )
            axes[1].set_title("t-SNE: Латентное пространство (без прокси)", fontsize=14, fontweight="bold")

        axes[1].axis("off")
        cbar = plt.colorbar(sc, ax=axes[1])
        cbar.set_label("Истинный RUL", rotation=270, labelpad=15)
    else:
        axes[1].text(0.5, 0.5, "t-SNE недоступен", ha="center", va="center", fontsize=14)
        axes[1].axis("off")

    zones = [(0, 20), (20, 50), (50, 100), (100, 150)]
    labels = ["0-20\n(Критично)", "20-50\n(Опасно)", "50-100\n(Средне)", ">100\n(Норма)"]
    colors = ["crimson", "darkorange", "gold", "mediumseagreen"]
    rmses = []

    for low, high in zones:
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        else:
            rmse = 0.0
        rmses.append(rmse)

    bars = axes[2].bar(labels, rmses, color=colors, edgecolor="k")
    axes[2].set_ylabel("RMSE", fontsize=12)
    axes[2].set_title("RMSE по зонам (Разложение ошибки)", fontsize=14, fontweight="bold")

    for bar in bars:
        yval = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.3,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"results_plot_{dataset_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_learning_curves(
    csv_path: str,
    dataset_name: str,
    save_dir: str,
    show: bool = False,
    mse_weight: float = 0.05,
    kl_weight: float = 1.0,
    lambda_order: float = 0.5,
) -> plt.Figure:
    """
    Отрисовка кривых обучения с учетом масштаба слагаемых лосса:
      - На основной левой оси откладываются Total Loss и взвешенные вклады в него
        (w_mse * MSE, w_kl * KL, lambda_order * Gram), чтобы их сумма соответствовала Total Loss и не сжимала масштаб.
      - На правой оси (twinx) отображается сырой ненормированный MSE_Loss (в циклах^2).
    """
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"Кривые обучения: {dataset_name}", fontsize=16, fontweight="bold", y=1.05)

    ax_loss = axes[0]
    if "Total_Loss" in df.columns:
        ax_loss.plot(df["Epoch"], df["Total_Loss"], label="Total Loss", linewidth=2.5, color="black", zorder=4)

        has_kl = "KL_Loss" in df.columns and df["KL_Loss"].notna().any() and df["KL_Loss"].sum() > 0
        has_order = "order_Loss" in df.columns and df["order_Loss"].notna().any() and df["order_Loss"].sum() > 0

        if has_kl or has_order or mse_weight < 1.0:
            if "MSE_Loss" in df.columns:
                w_mse_series = (
                    df["Weighted_MSE_Loss"]
                    if "Weighted_MSE_Loss" in df.columns
                    else df["MSE_Loss"] * mse_weight
                )
                ax_loss.plot(
                    df["Epoch"],
                    w_mse_series,
                    label=f"Weighted MSE ({mse_weight:g}·MSE)",
                    linestyle="-.",
                    linewidth=2.0,
                    color="dodgerblue",
                )
            if has_kl:
                w_kl_series = (
                    df["Weighted_KL_Loss"]
                    if "Weighted_KL_Loss" in df.columns
                    else df["KL_Loss"] * kl_weight
                )
                ax_loss.plot(
                    df["Epoch"],
                    w_kl_series,
                    label=f"KL Loss ({kl_weight:g}·KL)",
                    linestyle=":",
                    linewidth=2.2,
                    color="forestgreen",
                )
            if has_order:
                w_ord_series = (
                    df["Weighted_order_Loss"]
                    if "Weighted_order_Loss" in df.columns
                    else df["order_Loss"] * lambda_order
                )
                ax_loss.plot(
                    df["Epoch"],
                    w_ord_series,
                    label=f"Gram Order ({lambda_order:g}·Gram)",
                    linestyle="--",
                    linewidth=2.0,
                    color="crimson",
                )

            if "MSE_Loss" in df.columns:
                ax_mse_raw = ax_loss.twinx()
                ax_mse_raw.plot(
                    df["Epoch"],
                    df["MSE_Loss"],
                    label="Raw MSE (cycles²)",
                    linestyle="-",
                    linewidth=1.3,
                    color="steelblue",
                    alpha=0.45,
                )
                ax_mse_raw.set_ylabel("Raw MSE (cycles²)", fontsize=10, color="steelblue")
                ax_mse_raw.tick_params(axis="y", labelcolor="steelblue")
                ax_mse_raw.grid(False)
                lines1, labels1 = ax_loss.get_legend_handles_labels()
                lines2, labels2 = ax_mse_raw.get_legend_handles_labels()
                ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
            else:
                ax_loss.legend(loc="upper right", fontsize=9)
        else:
            if "MSE_Loss" in df.columns:
                ax_loss.plot(
                    df["Epoch"],
                    df["MSE_Loss"],
                    label="MSE Loss",
                    linestyle="-.",
                    linewidth=2.0,
                    color="dodgerblue",
                )
            ax_loss.legend(loc="upper right", fontsize=10)
    else:
        ax_loss.plot(df["Epoch"], df["MSE_Loss"], label="MSE Loss (Baseline)", linewidth=2.5, color="black")
        ax_loss.legend(loc="upper right", fontsize=10)

    ax_loss.set_title("Динамика слагаемых функции потерь (Loss)", fontsize=14, fontweight="bold")
    ax_loss.set_xlabel("Эпоха", fontsize=12)
    ax_loss.set_ylabel("Взвешенный Loss (масштаб Total Loss)", fontsize=11)
    ax_loss.grid(True, alpha=0.3)

    if "Val_RMSE" in df.columns and df["Val_RMSE"].notna().any() and (df["Val_RMSE"].astype(str) != "").any():
        axes[1].plot(df["Epoch"], df["Val_RMSE"], label="Validation RMSE", linewidth=2.5, color="mediumseagreen")
        if "Test_RMSE" in df.columns:
            axes[1].plot(
                df["Epoch"], df["Test_RMSE"], label="Test RMSE", linewidth=1.8, linestyle="--", color="darkorange", alpha=0.8
            )
        axes[1].set_title("Validation & Test RMSE", fontsize=14, fontweight="bold")
    else:
        axes[1].plot(df["Epoch"], df["Test_RMSE"], label="Test RMSE", linewidth=2.5, color="darkorange")
        axes[1].set_title("Test RMSE", fontsize=14, fontweight="bold")

    axes[1].set_xlabel("Эпоха", fontsize=12)
    axes[1].set_ylabel("RMSE (циклы)", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["Epoch"], df["Test_S_Score"], label="NASA S-Score", linewidth=2.5, color="purple")
    axes[2].set_title("Test S-Score (Log Scale)", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Эпоха", fontsize=12)
    axes[2].set_ylabel("NASA S-Score", fontsize=12)
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_baseline" if "baseline" in os.path.basename(csv_path) else ""
    save_path = os.path.join(save_dir, f"learning_curves_{dataset_name}{suffix}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_diagnostics(
    encoder: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    cfg: dict,
    dataset_name: str,
    epoch: int,
    save_dir: Optional[str] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Строит гистограмму активации прокси-аттракторов."""
    has_proxies = getattr(encoder, "head_type", "proxy") == "proxy"
    if not has_proxies:
        return None

    was_training = encoder.training
    encoder.eval()
    all_weights = []
    device = cfg["device"]

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            _, log_weights, _, _ = encoder(batch_x)
            if log_weights is not None:
                all_weights.append(torch.exp(log_weights).cpu())

    if was_training:
        encoder.train()

    if not all_weights:
        return None

    weights_cat = torch.cat(all_weights, dim=0)
    active_proxies = torch.argmax(weights_cat, dim=1).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(
        active_proxies,
        bins=np.arange(cfg["num_proxies"] + 1) - 0.5,
        rwidth=0.8,
        color="royalblue",
        edgecolor="black",
    )
    ax.set_xticks(range(0, cfg["num_proxies"], 2 if cfg["num_proxies"] > 12 else 1))
    ax.set_title(
        f"Активация прокси ({dataset_name}, K={cfg['num_proxies']}, Эпоха {epoch + 1})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Индекс прокси-аттрактора", fontsize=11)
    ax.set_ylabel("Количество срабатываний (argmax)", fontsize=11)

    out_dir = save_dir if save_dir is not None else cfg.get("save_dir", ".")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"diagnostics_{dataset_name}_epoch{epoch + 1}.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
