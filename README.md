# RUL Prediction — STAR + Proxy Metric Learning

Research code for Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan engine datasets (FD001–FD004). Implements a STAR Spatio-Temporal Transformer with learnable proxy attractors and Label Distribution Learning.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

**Option A: download automatically** (requires internet):

```bash
python scripts/download_data.py
```

**Option B: use your own copy** — pass `data_root` pointing to any directory with the 12 C-MAPSS `.txt` files:

```bash
PYTHONPATH=. python src/train.py data=fd001 data_root=/path/to/cmapss
```

### 3. Train

```bash
# FD001
PYTHONPATH=. python src/train.py data=fd001

# FD002
PYTHONPATH=. python src/train.py data=fd002

# With custom data path
PYTHONPATH=. python src/train.py data=fd001 data_root=/path/to/cmapss
```

## Inference

```bash
PYTHONPATH=. python src/inference.py data=fd001
```

When prompted, provide the path to your saved `.pth` checkpoint.

## Configuration

All parameters are configurable via Hydra command-line overrides:

```bash
PYTHONPATH=. python src/train.py data=fd001 trainer.epochs=30 model.num_proxies=16 model.temp=0.05
```

### Key parameters

| Parameter | Default (FD001) | Description |
|-----------|-----------------|-------------|
| `data.window_size` | 32 | Sliding window length |
| `data.batch_size` | 32 | Mini-batch size |
| `data.lr` | 2e-4 | Learning rate |
| `data.weight_decay` | 1e-4 | Weight decay for AdamW |
| `data.d_model` | 64 | Transformer embedding dimension |
| `data.nhead` | 1 | Multi-head attention heads |
| `data.num_scales` | 2 | Encoder scale stages |
| `data.ffn_dim` | 128 | Feedforward network dimension |
| `data.dropout` | 0.1 | Dropout rate |
| `model.num_proxies` | 12 | Number of learnable proxy attractors |
| `model.temp` | 0.02 | Softmax retrieval temperature |
| `model.emb_dim` | 64 | Metric embedding dimension |
| `trainer.epochs` | 20 | Total training epochs |
| `trainer.kl_weight` | 1.0 | KL-divergence loss weight |
| `trainer.mse_weight` | 0.05 | MSE loss weight |
| `trainer.ema_decay` | 0.99 | Exponential moving average decay |

## Outputs

- `checkpoints/best_proxy_FD00X.pth` — checkpoint of the best model
- `metrics_FD00X.csv` — logged epoch-by-epoch metrics
- `plots/` — diagnostic plots, loss curves, and t-SNE latent visualizations
- `runs/` — TensorBoard event logs (`tensorboard --logdir=runs/FD001`)
