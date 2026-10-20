# AGENTS.md

## What this is

Research code for Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan engine datasets (FD001–FD004).
Model: **STAR + Proxy Metric Learning** — Spatio-temporal Attention Transformer encoder with learnable proxy attractors and Label Distribution Learning loss (KL-divergence + MSE).

## Stack

Python 3.10+ · PyTorch · Hydra (config) · scikit-learn · pandas · numpy · seaborn · matplotlib

Install: `pip install -r requirements.txt`
Download data: `python scripts/download_data.py`

## Entry points

| Command | Purpose |
|---------|---------|
| `python src/train.py` | Train proxy metric learning model |
| `python src/inference.py` | Evaluate checkpoint (prompts for `.pth` path) |

### Training examples

```bash
PYTHONPATH=. python src/train.py data=fd001
PYTHONPATH=. python src/train.py data=fd002 trainer.epochs=30
PYTHONPATH=. python src/train.py data=fd001 data_root=/path/to/cmapss
```

## Data

C-MAPSS dataset files are not kept in the repository. Expected flat structure:

```
data/
  train_FD001.txt  test_FD001.txt  RUL_FD001.txt
  train_FD002.txt  test_FD002.txt  RUL_FD002.txt
  train_FD003.txt  test_FD003.txt  RUL_FD003.txt
  train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

Sensors used: 14 active sensors (`s1, s5, s6, s10, s16, s18, s19` dropped).
- FD001/FD003: `n_clusters=1` (global normalization)
- FD002/FD004: `n_clusters=6` (regime-specific normalization via KMeans)

## Architecture

`src/models/proxy_encoder.py`:
- `PatchEmbedDimWise` — dimension-wise patch projection per sensor with learnable positional embeddings
- `STARAttentionBlock` — dual spatio-temporal attention: temporal MHA per sensor + sensor MHA per timestep
- `PatchMerging` — hierarchical patch merging across scale stages
- `STAREncoder` — multi-scale encoder
- `ProxySREncoder` — full model: multi-scale last-patch pooling across layers, linear projection, cosine similarity against learnable proxy attractors, softmax weighting for continuous RUL prediction

`src/engine/loss.py`:
- `HybridProxyLoss` — weighted sum of MSE and KL-divergence
- Target distribution: Gaussian soft labels over discrete proxy RUL values with adaptive sigma

EMA tracking: `torch.optim.swa_utils.AveragedModel` with exponential moving average decay (0.99).

## Outputs

Hydra writes run artifacts to `artifacts/outputs/<timestamp>/`:
- `checkpoints/best_proxy_FD00X.pth` — best checkpoint by test RMSE
- `metrics_FD00X.csv` — per-epoch loss and evaluation metrics
- `plots/` — diagnostic plots, learning curves, t-SNE attractor map
- `runs/` — TensorBoard event files
