import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from src.data.cmapss_loader import prepare_cmapss
from src.data.windowing import extract_windows
from src.models.proxy_encoder import STARProxyModel
from src.utils.metrics import nasa_score, rmse
from src.utils.seed import set_seed


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() and cfg.trainer.device != "cpu" else "cpu")

    checkpoint_path = input("Path to checkpoint (.pth): ").strip()

    _, test_df, features = prepare_cmapss(
        cfg.data.train_path, cfg.data.test_path, cfg.data.rul_path,
        cfg.max_rul, cfg.data.n_clusters
    )
    X_test, y_test = extract_windows(test_df, features, cfg.data.window_size, is_test=True)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    encoder = STARProxyModel(
        window=cfg.data.window_size,
        n_sensors=len(features),
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

    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    encoder.load_state_dict(clean_state_dict)
    encoder.eval()

    with torch.no_grad():
        pred_rul, _, _, _ = encoder(X_test_tensor)
        preds = np.clip(pred_rul.cpu().numpy(), 0, cfg.max_rul)

    test_rmse = rmse(y_test, preds)
    test_score = nasa_score(y_test, preds)

    print(f"Dataset:    {cfg.data.dataset_name}")
    print(f"RMSE:       {test_rmse:.4f}")
    print(f"NASA Score: {test_score:.2f}")


if __name__ == "__main__":
    main()
