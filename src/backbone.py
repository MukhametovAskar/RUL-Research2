import math
from typing import List

import torch
import torch.nn as nn


class PatchEmbedDimWise(nn.Module):
    """Поканальное (dimension-wise) разбиение временного окна каждого датчика на патчи с обучаемым позиционным кодированием."""

    def __init__(self, window: int, n_sensors: int, patch_size: int, d_model: int, pos_learnable: bool = True):
        super().__init__()
        self.window = window
        self.P = patch_size
        self.n_patches = math.ceil(window / patch_size)
        self.d_model = d_model
        self.patch_proj = nn.Linear(self.P, d_model, bias=True)
        if pos_learnable:
            self.pos_embed = nn.Parameter(torch.zeros(n_sensors, self.n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, S = x.shape
        pad_len = (self.n_patches * self.P) - W
        if pad_len > 0:
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
        x = x.view(B, self.n_patches, self.P, S).permute(0, 3, 1, 2).contiguous()
        emb_flat = self.patch_proj(x.view(B * S * self.n_patches, self.P))
        emb = emb_flat.view(B, S, self.n_patches, self.d_model)
        if self.pos_embed is not None:
            emb = emb + self.pos_embed.unsqueeze(0)
        return emb


class STARAttentionBlock(nn.Module):
    """Двойной пространственно-временной блок внимания (Temporal MHA внутри датчика + Sensor MHA между датчиками)."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.temporal_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.temporal_norm1 = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.temporal_norm2 = nn.LayerNorm(d_model)

        self.sensor_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.sensor_norm1 = nn.LayerNorm(d_model)
        self.sensor_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.sensor_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, T, d = x.shape
        res = x
        x_flat = x.view(B * S, T, d)
        temp_out, _ = self.temporal_mha(x_flat, x_flat, x_flat)
        x = self.temporal_norm1(res + temp_out.view(B, S, T, d))
        x = self.temporal_norm2(x + self.temporal_ffn(x))

        res = x
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * T, S, d)
        sensor_out, _ = self.sensor_mha(x_flat, x_flat, x_flat)
        x = self.sensor_norm1(res + sensor_out.view(B, T, S, d).permute(0, 2, 1, 3))
        x = self.sensor_norm2(x + self.sensor_ffn(x))
        return x


class PatchMerging(nn.Module):
    """Иерархическое слияние соседних временных патчей между масштабами."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, T, d = x.shape
        if T <= 1:
            return x
        if T % 2 == 1:
            x = x[:, :, :-1, :]
            T = T - 1
        left = x[:, :, 0::2, :]
        right = x[:, :, 1::2, :]
        merged = torch.cat([left, right], dim=-1)
        return self.proj(merged)


class STAREncoder(nn.Module):
    """Многомасштабный иерархический энкодер STAR."""

    def __init__(
        self,
        n_scales: int,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float,
        n_layers_per_scale: int = 2,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [STARAttentionBlock(d_model, nhead, ffn_dim, dropout) for _ in range(n_layers_per_scale)]
                )
                for _ in range(n_scales)
            ]
        )
        self.patch_merging = nn.ModuleList([PatchMerging(d_model) for _ in range(n_scales - 1)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        cur = x
        for i in range(self.n_scales):
            for layer in self.layers[i]:
                cur = layer(cur)
            features.append(cur)
            if i < self.n_scales - 1:
                cur = self.patch_merging[i](cur)
        return features
