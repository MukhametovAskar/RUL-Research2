import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridProxyLoss(nn.Module):
    def __init__(self, proxy_ruls, num_proxies, max_rul, kl_weight=1.0, mse_weight=0.05):
        super().__init__()
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        step = max_rul / (num_proxies - 1)
        self.sigma = 0.425 * step
        self.register_buffer('proxy_ruls', proxy_ruls.clone())

    def get_target_probs(self, true_rul):
        dist_sq = (true_rul.unsqueeze(1) - self.proxy_ruls.unsqueeze(0)) ** 2
        return F.softmax(-dist_sq / (2 * self.sigma ** 2), dim=1)

    def forward(self, pred_rul, log_weights, true_rul):
        loss_mse = F.mse_loss(pred_rul, true_rul)
        target_probs = self.get_target_probs(true_rul)
        loss_kl = F.kl_div(log_weights, target_probs, reduction='batchmean')
        total_loss = (self.mse_weight * loss_mse) + (self.kl_weight * loss_kl)
        return total_loss, loss_mse, loss_kl
