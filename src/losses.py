import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridProxyLoss(nn.Module):
    def __init__(
        self,
        proxy_ruls: torch.Tensor,
        num_proxies: int,
        max_rul: int,
        kl_weight: float = 1.0,
        mse_weight: float = 0.05,
        lambda_order: float = 0.5,
        use_proxy_1k: bool = True,
        use_residual: bool = True,
        head_type: str = "proxy",
    ):
        super().__init__()
        self.register_buffer("proxy_ruls", proxy_ruls.detach().clone().float())
        self.num_proxies = num_proxies
        self.max_rul = max_rul
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        self.lambda_order = lambda_order
        self.use_proxy_1k = use_proxy_1k
        self.use_residual = use_residual
        self.head_type = head_type

        self.step = (proxy_ruls[1] - proxy_ruls[0]).item()
        self.sigma = 0.5 * self.step

        dist = (proxy_ruls[:, None] - proxy_ruls[None, :]).abs()
        max_dist = (proxy_ruls[-1] - proxy_ruls[0]).item()
        target_angles = math.pi * (dist / max_dist)
        self.register_buffer("gram_target", torch.cos(target_angles))

    def get_target_probs(self, true_rul: torch.Tensor) -> torch.Tensor:
        dist_sq = (true_rul.unsqueeze(1) - self.proxy_ruls.unsqueeze(0)) ** 2
        return F.softmax(-dist_sq / (2 * self.sigma ** 2), dim=1)

    def proxy_order_reg(self, proxies_norm: torch.Tensor) -> torch.Tensor:
        gram = torch.matmul(proxies_norm, proxies_norm.T)
        return F.mse_loss(gram, self.gram_target)

    def _aggregate(
        self,
        per_sample_loss: torch.Tensor,
        target_probs: torch.Tensor,
        proxy_mass: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_proxy_1k and active.any():
            return ((target_probs[:, active] * per_sample_loss.unsqueeze(1)).sum(dim=0) / proxy_mass[active]).mean()
        return per_sample_loss.mean()

    def forward(
        self,
        pred_rul: torch.Tensor,
        log_weights: Optional[torch.Tensor],
        true_rul: torch.Tensor,
        proxies_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_probs = self.get_target_probs(true_rul)
        proxy_mass = target_probs.sum(dim=0)
        active = proxy_mass > 0.1

        mse_pred_per_sample = (pred_rul - true_rul) ** 2
        loss_mse_res = self._aggregate(mse_pred_per_sample, target_probs, proxy_mass, active)

        if self.head_type == "proxy" and log_weights is not None:
            weights = torch.exp(log_weights)
            proxy_rul = torch.sum(weights * self.proxy_ruls, dim=1)

            mse_proxy_per_sample = (proxy_rul - true_rul) ** 2
            kl_per_sample = F.kl_div(log_weights, target_probs, reduction="none").sum(dim=-1)

            loss_mse_proxy = self._aggregate(mse_proxy_per_sample, target_probs, proxy_mass, active)
            loss_kl = self._aggregate(kl_per_sample, target_probs, proxy_mass, active)

            if self.use_residual:
                total_loss = (
                    (self.mse_weight * loss_mse_proxy)
                    + (self.mse_weight * loss_mse_res)
                    + (self.kl_weight * loss_kl)
                )
            else:
                total_loss = (self.mse_weight * loss_mse_proxy) + (self.kl_weight * loss_kl)
        else:
            loss_mse_proxy = loss_mse_res
            loss_kl = torch.tensor(0.0, device=pred_rul.device, dtype=pred_rul.dtype)
            total_loss = self.mse_weight * loss_mse_res

        if proxies_norm is not None and self.lambda_order > 0:
            loss_order = self.proxy_order_reg(proxies_norm)
            total_loss = total_loss + self.lambda_order * loss_order
        else:
            loss_order = torch.tensor(0.0, device=pred_rul.device, dtype=pred_rul.dtype)

        return total_loss, loss_mse_res, loss_kl, loss_order, loss_mse_proxy


def build_criterion(cfg: dict, model: nn.Module) -> HybridProxyLoss:
    proxy_ruls = getattr(model, "proxy_ruls", None)
    if proxy_ruls is None:
        proxy_ruls = torch.linspace(-10.0, cfg["max_rul"] + 10.0, cfg["num_proxies"])

    return HybridProxyLoss(
        proxy_ruls=proxy_ruls,
        num_proxies=cfg["num_proxies"],
        max_rul=cfg["max_rul"],
        kl_weight=cfg.get("kl_weight", 1.0),
        mse_weight=cfg.get("mse_weight", 0.05),
        lambda_order=cfg.get("lambda_order", 0.5),
        use_proxy_1k=cfg.get("use_proxy_1k", True),
        use_residual=cfg.get("use_residual", True),
        head_type=cfg.get("head_type", "proxy"),
    )
