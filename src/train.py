import csv
import os
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.cmapss_loader import prepare_cmapss
from src.data.windowing import CMAPSSDataset, extract_windows
from src.engine.evaluator import evaluate
from src.engine.loss import HybridProxyLoss
from src.engine.trainer import train_one_epoch
from src.models.proxy_encoder import STARProxyModel
from src.utils.metrics import nasa_score, rmse
from src.utils.plots import plot_diagnostics, plot_learning_curves, plot_results
from src.utils.seed import set_seed


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() and cfg.trainer.device != "cpu" else "cpu")

    train_df, test_df, features = prepare_cmapss(
        cfg.data.train_path, cfg.data.test_path, cfg.data.rul_path,
        cfg.max_rul, cfg.data.n_clusters
    )

    X_train, y_train = extract_windows(train_df, features, cfg.data.window_size, is_test=False)
    X_test, y_test = extract_windows(test_df, features, cfg.data.window_size, is_test=True)

    train_loader = DataLoader(
        CMAPSSDataset(X_train, y_train),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=True
    )
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    n_sensors = len(features)
    encoder = STARProxyModel(
        window=cfg.data.window_size,
        n_sensors=n_sensors,
        emb_dim=cfg.model.emb_dim,
        num_proxies=cfg.model.num_proxies,
        max_rul=cfg.max_rul,
        temp=cfg.model.temp,
        dropout=cfg.data.dropout,
        d_model=cfg.data.d_model,
        nhead=cfg.data.nhead,
        num_scales=cfg.data.num_scales,
        ffn_dim=cfg.data.ffn_dim,
        patch_size=cfg.model.patch_size,
        pos_learnable=cfg.model.pos_learnable,
    ).to(device)

    criterion = HybridProxyLoss(
        proxy_ruls=encoder.proxy_ruls,
        num_proxies=cfg.model.num_proxies,
        max_rul=cfg.max_rul,
        kl_weight=cfg.trainer.kl_weight,
        mse_weight=cfg.trainer.mse_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.data.lr, weight_decay=cfg.data.weight_decay)
    ema_encoder = AveragedModel(encoder, multi_avg_fn=get_ema_multi_avg_fn(cfg.trainer.ema_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.trainer.epochs, eta_min=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    best_model_path = f"checkpoints/best_proxy_{cfg.data.dataset_name}.pth"
    best_rmse = float("inf")
    best_preds = None

    writer = SummaryWriter(log_dir=f"runs/{cfg.data.dataset_name}")
    csv_path = f"metrics_{cfg.data.dataset_name}.csv"

    with open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Total_Loss", "MSE_Loss", "KL_Loss", "Test_RMSE", "Test_S_Score"])

        for epoch in range(cfg.trainer.epochs):
            avg_loss, avg_mse, avg_kl = train_one_epoch(
                encoder, ema_encoder, train_loader, optimizer, criterion, device
            )
            scheduler.step()

            cur_rmse, cur_score, preds = evaluate(ema_encoder, X_test_tensor, y_test, cfg.max_rul)
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"[{cfg.data.dataset_name}] Epoch [{epoch + 1:02d}/{cfg.trainer.epochs}] "
                f"| Loss={avg_loss:.4f} | MSE={avg_mse:.4f} | KL={avg_kl:.4f} "
                f"| RMSE={cur_rmse:.4f} | Score={cur_score:.2f} | LR={lr:.6f}"
            )

            writer.add_scalar("Loss/Total", avg_loss, epoch)
            writer.add_scalar("Loss/MSE", avg_mse, epoch)
            writer.add_scalar("Loss/KL", avg_kl, epoch)
            writer.add_scalar("Metrics/Test_RMSE", cur_rmse, epoch)
            writer.add_scalar("Metrics/Test_S_Score", cur_score, epoch)
            writer.add_scalar("Hyperparams/LR", lr, epoch)

            csv_writer.writerow([
                epoch + 1,
                round(avg_loss, 4),
                round(avg_mse, 4),
                round(avg_kl, 4),
                round(cur_rmse, 4),
                round(cur_score, 2),
            ])
            csv_file.flush()

            if epoch == 2 or epoch == cfg.trainer.epochs - 1:
                plot_diagnostics(
                    encoder, criterion, train_loader, device,
                    cfg.model.num_proxies, cfg.max_rul, cfg.model.temp,
                    cfg.data.dataset_name, epoch, save_dir="plots"
                )

            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                best_preds = preds.copy()
                torch.save(ema_encoder.state_dict(), best_model_path)
                writer.add_scalar("Best/RMSE", best_rmse, epoch)
                writer.add_scalar("Best/S_Score", cur_score, epoch)

            if cfg.trainer.save_epoch is not None:
                save_epochs = [cfg.trainer.save_epoch] if isinstance(cfg.trainer.save_epoch, int) else list(cfg.trainer.save_epoch)
                if (epoch + 1) in save_epochs:
                    epoch_model_path = f"checkpoints/epoch{epoch + 1}_proxy_{cfg.data.dataset_name}.pth"
                    torch.save(ema_encoder.state_dict(), epoch_model_path)
                    print(f"  -> Saved epoch {epoch + 1} model to {epoch_model_path}")

    last_model_path = f"checkpoints/last_proxy_{cfg.data.dataset_name}.pth"
    torch.save(ema_encoder.state_dict(), last_model_path)
    print(f"  -> Saved last model to {last_model_path}")

    plot_learning_curves(csv_path, cfg.data.dataset_name, save_dir="plots")

    ema_encoder.load_state_dict(torch.load(best_model_path, map_location=device))
    ema_encoder.eval()

    underlying = ema_encoder.module if hasattr(ema_encoder, "module") else ema_encoder
    proxies_np = F.normalize(underlying.proxies, p=2, dim=1).detach().cpu().numpy()
    proxy_ruls_np = underlying.proxy_ruls.cpu().numpy()

    eval_loader = DataLoader(CMAPSSDataset(X_train, y_train), batch_size=cfg.data.batch_size, shuffle=False)
    train_emb_list = []
    with torch.no_grad():
        for batch_x, _ in eval_loader:
            _, _, emb, _ = ema_encoder(batch_x.to(device))
            train_emb_list.append(emb.cpu().numpy())
    train_embeddings = np.vstack(train_emb_list)

    sample_size = min(3000, len(train_embeddings))
    indices = np.random.choice(len(train_embeddings), sample_size, replace=False)
    fig = plot_results(
        y_test, best_preds, train_embeddings[indices], y_train[indices],
        proxies_np, proxy_ruls_np, cfg.data.dataset_name, save_dir="plots"
    )
    writer.add_figure("Results/Metric_Learning", fig, global_step=cfg.trainer.epochs)
    writer.close()

    print(f"\nDone. Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
