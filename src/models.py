import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import PatchEmbedDimWise, STAREncoder


class PatchAttnPool(nn.Module):
    def __init__(self, d_model: int, num_patches: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.pos_bias = nn.Parameter(torch.linspace(-3.0, 0.0, num_patches))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        P = h.size(2)
        s = self.score(self.norm(h)).squeeze(-1) + self.pos_bias[:P]
        a = F.softmax(s, dim=-1).unsqueeze(-1)
        return (a * h).sum(dim=2)


class ProxySREncoder(nn.Module):
    def __init__(
        self,
        window: int,
        n_sensors: int,
        emb_dim: int,
        num_proxies: int,
        max_rul: int,
        device: torch.device,
        temp: float = 0.02,
        dropout: float = 0.1,
        d_model: int = 64,
        nhead: int = 2,
        num_scales: int = 2,
        ffn_dim: int = 128,
        patch_size: int = 4,
        sensor_pool: str = "mean",
        head_type: str = "proxy",
        use_residual: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_sensors = n_sensors
        self.sensor_pool = sensor_pool
        self.head_type = head_type
        self.use_residual = use_residual

        self.patch_embed = PatchEmbedDimWise(window, n_sensors, patch_size, d_model, pos_learnable=True)
        self.encoder = STAREncoder(num_scales, d_model, nhead, ffn_dim, dropout=dropout, n_layers_per_scale=2)

        cur_p = math.ceil(window / patch_size)
        patch_counts = []
        for i in range(num_scales):
            patch_counts.append(cur_p)
            if i < num_scales - 1 and cur_p > 1:
                cur_p = cur_p // 2

        self.patch_pools = nn.ModuleList(
            [PatchAttnPool(d_model, num_patches=p_cnt) for p_cnt in patch_counts]
        )

        if self.sensor_pool == "hybrid":
            self.sensor_scores = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, 1),
                    )
                    for _ in range(num_scales)
                ]
            )
            for m in range(num_scales):
                nn.init.zeros_(self.sensor_scores[m][-1].weight)
                nn.init.zeros_(self.sensor_scores[m][-1].bias)
        else:
            self.sensor_scores = None

        in_features = d_model * num_scales
        self.fc = nn.Linear(in_features, emb_dim)

        self.temp = temp
        self.device = device

        # Сетка опорных значений RUL [-10, max_rul + 10] хранится как буфер для всех моделей
        self.register_buffer("proxy_ruls", torch.linspace(-10.0, max_rul + 10.0, num_proxies))
        self.proxy_step = (self.proxy_ruls[1] - self.proxy_ruls[0]).item()

        if self.head_type == "proxy":
            # Сначала инициализируем прокси, чтобы сохранить точный RNG seed как в ноутбуке
            self.proxies = nn.Parameter(torch.randn(num_proxies, emb_dim))
            nn.init.orthogonal_(self.proxies)

            if self.use_residual:
                self.res_head = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Dropout(dropout),
                    nn.Linear(in_features, emb_dim // 2),
                    nn.GELU(),
                    nn.Linear(emb_dim // 2, 1),
                )
                nn.init.zeros_(self.res_head[-1].weight)
                nn.init.zeros_(self.res_head[-1].bias)
            else:
                self.res_head = None
            self.reg_head = None
        elif self.head_type == "linear":
            self.proxies = None
            self.res_head = None
            self.reg_head = nn.Linear(emb_dim, 1)
        else:
            raise ValueError(f"Unsupported head_type '{self.head_type}'. Expected 'proxy' or 'linear'.")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        emb = self.patch_embed(x)
        enc_feats = self.encoder(emb)

        pooled_feats = []
        for i, feat in enumerate(enc_feats):
            h = self.patch_pools[i](feat)

            if self.sensor_pool == "hybrid" and self.sensor_scores is not None:
                attn_w = F.softmax(self.sensor_scores[i](h), dim=1)
                attn_pooled = (h * attn_w).sum(dim=1)
                mean_pooled = h.mean(dim=1)
                pooled = 0.5 * mean_pooled + 0.5 * attn_pooled
            else:
                pooled = h.mean(dim=1)

            pooled_feats.append(pooled)

        concat_feat = torch.cat(pooled_feats, dim=-1)
        raw_emb = self.fc(concat_feat)

        if self.head_type == "proxy":
            emb_norm = F.normalize(raw_emb, p=2, dim=1)
            proxies_norm = F.normalize(self.proxies, p=2, dim=1)

            logits = torch.matmul(emb_norm, proxies_norm.T) / self.temp
            weights = F.softmax(logits, dim=1)
            log_weights = F.log_softmax(logits, dim=1)

            proxy_rul = torch.sum(weights * self.proxy_ruls, dim=1)

            if self.use_residual and self.res_head is not None:
                delta_rul = self.proxy_step * torch.tanh(self.res_head(concat_feat).squeeze(-1))
                pred_rul = proxy_rul.detach() + delta_rul
            else:
                pred_rul = proxy_rul

            return pred_rul, log_weights, emb_norm, proxies_norm

        elif self.head_type == "linear":
            pred_rul = self.reg_head(raw_emb).squeeze(-1)
            emb_norm = F.normalize(raw_emb, p=2, dim=1)
            return pred_rul, None, emb_norm, None

        raise ValueError(f"Unsupported head_type: '{self.head_type}'")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["proxies", "pos_embed", "pos_bias", "norm"]) or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def build_model(cfg: dict, n_sensors: int) -> ProxySREncoder:
    device = cfg["device"]
    model = ProxySREncoder(
        window=cfg["seq_len"],
        n_sensors=n_sensors,
        emb_dim=cfg["emb_dim"],
        num_proxies=cfg["num_proxies"],
        max_rul=cfg["max_rul"],
        device=device,
        temp=cfg["temp"],
        dropout=cfg["dropout"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_scales=cfg["num_scales"],
        ffn_dim=cfg["ffn_dim"],
        patch_size=cfg["patch_size"],
        sensor_pool=cfg.get("sensor_pool", "mean"),
        head_type=cfg.get("head_type", "proxy"),
        use_residual=cfg.get("use_residual", True),
    ).to(device)
    return model
