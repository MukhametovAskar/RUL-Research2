import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE


def plot_results(y_true, y_pred, train_emb, y_train_subset, proxies_emb, proxy_ruls, dataset_name, save_dir):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"Proxy-Anchor Metric Learning: {dataset_name}", fontsize=18, fontweight="bold", y=1.05)

    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor="k", c="dodgerblue", s=30)
    axes[0].plot([0, 130], [0, 130], "r--", linewidth=2, label="Ideal (True = Pred)")
    axes[0].set_xlabel("True RUL", fontsize=12)
    axes[0].set_ylabel("Predicted RUL", fontsize=12)
    axes[0].set_title("True vs Predicted RUL (Test Set)", fontsize=14, fontweight="bold")
    axes[0].legend()

    combined_emb = np.vstack([train_emb, proxies_emb])
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    combined_2d = tsne.fit_transform(combined_emb)

    num_proxies = len(proxy_ruls)
    emb_2d = combined_2d[:-num_proxies]
    proxies_2d = combined_2d[-num_proxies:]

    sc = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_train_subset, cmap="coolwarm_r", s=15, alpha=0.5)
    axes[1].scatter(proxies_2d[:, 0], proxies_2d[:, 1], c=proxy_ruls, cmap="coolwarm_r", marker="*", s=400, edgecolor="black", linewidth=1.5, zorder=5)
    axes[1].set_title("t-SNE: Data and Proxy Attractors", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label("True RUL", rotation=270, labelpad=15)

    zones = [(0, 20), (20, 50), (50, 100), (100, 150)]
    labels = ["0-20\n(Critical)", "20-50\n(Warning)", "50-100\n(Moderate)", ">100\n(Normal)"]
    colors = ["crimson", "darkorange", "gold", "mediumseagreen"]
    rmses = []

    for low, high in zones:
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            rmse_val = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
        else:
            rmse_val = 0
        rmses.append(rmse_val)

    bars = axes[2].bar(labels, rmses, color=colors, edgecolor="k")
    axes[2].set_ylabel("RMSE", fontsize=12)
    axes[2].set_title("RMSE by Zone", fontsize=14, fontweight="bold")

    for bar in bars:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"results_plot_{dataset_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_learning_curves(csv_path, dataset_name, save_dir):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"Learning Curves: {dataset_name}", fontsize=16, fontweight="bold", y=1.05)

    if "Total_Loss" in df.columns:
        axes[0].plot(df["Epoch"], df["Total_Loss"], label="Total Loss", linewidth=2.5, color="black")
        if "MSE_Loss" in df.columns:
            axes[0].plot(df["Epoch"], df["MSE_Loss"], label="MSE Loss", linestyle="-.", color="dodgerblue")
        if "KL_Loss" in df.columns:
            axes[0].plot(df["Epoch"], df["KL_Loss"], label="KL Loss", linestyle=":", color="forestgreen")

    axes[0].set_title("Loss Dynamics", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "Val_RMSE" in df.columns and df["Val_RMSE"].notna().any():
        axes[1].plot(df["Epoch"], df["Val_RMSE"], label="Validation RMSE", linewidth=2.5, color="mediumseagreen")
        axes[1].set_title("Validation RMSE", fontsize=14)
    elif "Test_RMSE" in df.columns:
        axes[1].plot(df["Epoch"], df["Test_RMSE"], label="Test RMSE", linewidth=2.5, color="darkorange")
        axes[1].set_title("Test RMSE", fontsize=14)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if "Test_S_Score" in df.columns:
        axes[2].plot(df["Epoch"], df["Test_S_Score"], label="NASA S-Score", linewidth=2.5, color="purple")
        axes[2].set_title("Test S-Score (Log Scale)", fontsize=14)
        axes[2].set_xlabel("Epoch", fontsize=12)
        axes[2].set_yscale("log")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"learning_curves_{dataset_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_diagnostics(encoder, criterion, data_loader, device, num_proxies, max_rul, temp, dataset_name, epoch, save_dir):
    encoder.eval()
    all_weights = []
    all_target_probs = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            _, log_weights, _, _ = encoder(batch_x)
            weights = torch.exp(log_weights)
            target_probs = criterion.get_target_probs(batch_y)
            all_weights.append(weights.cpu())
            all_target_probs.append(target_probs.cpu())

    weights_cat = torch.cat(all_weights, dim=0)
    targets_cat = torch.cat(all_target_probs, dim=0)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"Metric Space Diagnostics: {dataset_name} | Epoch {epoch + 1}", fontsize=16, fontweight="bold", y=1.05)

    active_proxies = torch.argmax(weights_cat, dim=1).numpy()
    axes[0].hist(active_proxies, bins=np.arange(num_proxies + 1) - 0.5, rwidth=0.8, color="royalblue", edgecolor="black")
    axes[0].set_xticks(range(num_proxies))
    axes[0].set_title(f"Proxy Activation (K={num_proxies})", fontsize=12)
    axes[0].set_xlabel("Proxy Index", fontsize=11)
    axes[0].set_ylabel("Activation Count (argmax)", fontsize=11)

    test_ruls = torch.linspace(0, max_rul, 500).to(device)
    test_targets = criterion.get_target_probs(test_ruls).cpu().numpy()
    for i in range(num_proxies):
        axes[1].plot(test_ruls.cpu().numpy(), test_targets[:, i], linewidth=2.5)
    axes[1].set_title(f"Gaussian Overlap (sigma={criterion.sigma:.2f})", fontsize=12)
    axes[1].set_xlabel("RUL", fontsize=11)
    axes[1].set_ylabel("Target Prob", fontsize=11)

    entropy_target = -torch.sum(targets_cat * torch.log(targets_cat + 1e-9), dim=1).numpy()
    entropy_pred = -torch.sum(weights_cat * torch.log(weights_cat + 1e-9), dim=1).numpy()

    axes[2].scatter(entropy_target, entropy_pred, alpha=0.4, color="darkorange", s=20)
    min_val = min(np.min(entropy_target), np.min(entropy_pred))
    max_val = max(np.max(entropy_target), np.max(entropy_pred))

    axes[2].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2.5, label="Ideal")
    axes[2].set_title(f"Entropy Calibration (temp={temp})", fontsize=12)
    axes[2].set_xlabel("Target Entropy", fontsize=11)
    axes[2].set_ylabel("Predicted Entropy", fontsize=11)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"diagnostics_{dataset_name}_epoch{epoch + 1}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    encoder.train()
    return fig
