import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedDimWise(nn.Module):
    def __init__(self, window, n_sensors, patch_size, d_model, pos_learnable=True):
        super().__init__()
        self.P = patch_size
        self.n_patches = math.ceil(window / patch_size)
        self.patch_proj = nn.Linear(patch_size, d_model, bias=True)
        if pos_learnable:
            self.pos_embed = nn.Parameter(torch.zeros(n_sensors, self.n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(self, x):
        B, W, S = x.shape
        pad_len = (self.n_patches * self.P) - W
        if pad_len > 0:
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
        x = x.view(B, self.n_patches, self.P, S).permute(0, 3, 1, 2).contiguous()
        emb_flat = self.patch_proj(x.view(B * S * self.n_patches, self.P))
        emb = emb_flat.view(B, S, self.n_patches, -1)
        if self.pos_embed is not None:
            emb = emb + self.pos_embed.unsqueeze(0)
        return emb


class STARAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim=128, dropout=0.1):
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

    def forward(self, x):
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
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
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
    def __init__(self, n_scales, d_model, nhead, ffn_dim, dropout, n_layers_per_scale=2):
        super().__init__()
        self.n_scales = n_scales
        self.layers = nn.ModuleList([
            nn.ModuleList([STARAttentionBlock(d_model, nhead, ffn_dim, dropout) for _ in range(n_layers_per_scale)])
            for _ in range(n_scales)
        ])
        self.patch_merging = nn.ModuleList([PatchMerging(d_model) for _ in range(n_scales - 1)])

    def forward(self, x):
        features = []
        cur = x
        for i in range(self.n_scales):
            for layer in self.layers[i]:
                cur = layer(cur)
            features.append(cur)
            if i < self.n_scales - 1:
                cur = self.patch_merging[i](cur)
        return features


class ProxySREncoder(nn.Module):
    def __init__(self, window, n_sensors, emb_dim, num_proxies, max_rul, temp=0.1, dropout=0.1,
                 d_model=64, nhead=2, num_scales=2, ffn_dim=128, patch_size=4, pos_learnable=True):
        super().__init__()
        self.patch_embed = PatchEmbedDimWise(window, n_sensors, patch_size, d_model, pos_learnable=pos_learnable)
        self.encoder = STAREncoder(num_scales, d_model, nhead, ffn_dim, dropout=dropout, n_layers_per_scale=2)
        self.fc = nn.Linear(d_model * num_scales, emb_dim)
        self.temp = temp
        self.proxies = nn.Parameter(torch.randn(num_proxies, emb_dim))
        nn.init.orthogonal_(self.proxies)
        self.register_buffer("proxy_ruls", torch.linspace(0, max_rul, num_proxies))

    def forward(self, x):
        emb = self.patch_embed(x)
        enc_feats = self.encoder(emb)
        pooled_feats = []
        for feat in enc_feats:
            pooled = feat[:, :, -1, :].mean(dim=1)
            pooled_feats.append(pooled)
        concat_feat = torch.cat(pooled_feats, dim=-1)
        raw_emb = self.fc(concat_feat)

        emb_norm = F.normalize(raw_emb, p=2, dim=1)
        proxies_norm = F.normalize(self.proxies, p=2, dim=1)

        sim = torch.matmul(emb_norm, proxies_norm.T)
        weights = F.softmax(sim / self.temp, dim=1)
        log_weights = F.log_softmax(sim / self.temp, dim=1)
        pred_rul = torch.sum(weights * self.proxy_ruls, dim=1)

        return pred_rul, log_weights, emb_norm, proxies_norm
