# STAR-Proxy RUL — Multi-Seed Ablation Study

Research repository for Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan engine datasets (`FD001`, `FD002`, `FD003`, `FD004`) and a 5-seed Ablation Study (`42, 43, 44, 45, 50`).

## Architecture Overview

- **Backbone (`STAREncoder`)**: Dimension-wise patch embedding (`PatchEmbedDimWise`), hierarchical spatio-temporal attention blocks (`STARAttentionBlock`), multi-scale patch merging (`PatchMerging`), and positional temporal pooling (`PatchAttnPool`).
- **Spherical Proxy-Head + Decoupled Residual Head (`ProxySREncoder`)**:
  - $L_2$-normalized latent space and $K=16$ orthogonal learnable proxy attractors on the extended grid $[-10, 135]$ with sharp temperature $\tau=0.02$.
  - Two-stage coarse-to-fine prediction: coarse spherical proxy expectation $\hat{y}_{\text{proxy}}$ plus bounded local residual refinement $\Delta\hat{y} \in [-\Delta, +\Delta]$ via `proxy_rul.detach() + delta_rul`.
- **Loss Function (`HybridProxyLoss`)**:
  - **Proxy-Centric $1/K$ Aggregation**: Normalizes sample errors within each active proxy attractor ($m_k > 0.1$) and averages across active proxies ($1/K$), preventing the $\text{RUL}=125$ constant shelf from dominating transition zones ($\text{RUL} \in [50, 100]$).
  - **Label Distribution Learning (LDL)**: Soft Gaussian target distributions ($\sigma = 0.5 \cdot \text{step}$) trained with KL-divergence ($\lambda_{\text{KL}}=1.0$).
  - **Cosine Gram Order Regularizer**: Enforces geodesic ordering of proxies along a hemisphere ($\cos(\pi \cdot |r_i - r_j| / (r_{\max} - r_{\min}))$, $\lambda_{\text{order}}=0.5$).
- **Leak-Free Validation & Regime-Specific Normalization (`src/data.py`)**:
  - For validation runs (`--use_validation`), engine units are split into `train_units` and `val_units` **before** fitting KMeans ($K=6$ clusters on `FD002`/`FD004`, $K=1$ on `FD001`/`FD003`) and computing Min-Max scaling statistics. Validation and test sets are transformed strictly via `.predict()` and training extrema.

---

## Ablation Study Experiments (`00` – `07`)

| ID | Name | Description |
| :--- | :--- | :--- |
| `00_full_model` | **0. Full Proposed Model** | `STAREncoder` + Spherical Proxy-Head + LDL (`KL-Loss`) + Cosine Gram Order + Decoupled Residual Head + Proxy-Centric $1/K$ Loss + Regime-Specific Norm (`FD002`/`FD004`) |
| `01_linear_head_standard_mse` | **1. Linear Head + Standard MSE ($1/B$)** | Replaces Proxy-Head and Residual-Head with direct `Linear -> RUL` trained with standard batch-mean MSE ($1/B$) |
| `02_linear_head_balanced_mse_1k` | **2. Linear Head + Balanced MSE ($1/K$)** | Direct `Linear -> RUL` without proxies, trained with balanced $1/K$ grid-weighted MSE loss |
| `03_wo_ldl_kl_loss` | **3. w/o LDL / KL-Loss (`kl_weight = 0`)** | Keeps Proxy-Head & Residual-Head, disables KL-Loss (`kl_weight=0.0`), training on MSE + Gram Order |
| `04_wo_proxy_1k_aggregation` | **4. w/o Proxy-Centric $1/K$ Aggregation ($1/B$)** | Full proxy architecture with standard batch-mean ($1/B$) aggregation instead of $1/K$ active-proxy averaging |
| `05_wo_residual_proxy_head` | **5. w/o Residual Proxy Head** | Disables local residual correction $\Delta\text{RUL}$ (`pred_rul = proxy_rul`, coarse proxies only) |
| `06_wo_cosine_gram_regularizer` | **6. w/o Cosine Gram Regularizer** | Disables Cosine Gram matrix ordering regularizer (`lambda_order=0.0`) |
| `07_wo_regime_normalization` | **7. w/o Regime-Specific Normalization** | Uses Global Normalization ($1$ cluster) instead of $6$-cluster KMeans on multi-regime `FD002`/`FD004` (duplicates `00_full_model` on single-regime `FD001`/`FD003`) |

---

## Default Hyperparameters

- **Seeds (5 seeds):** `42, 43, 44, 45, 50` (default single-run seed: `50`)
- **Common parameters:** `max_rul=125`, `num_proxies=16`, `temp=0.02`, `ema_decay=0.99`, `kl_weight=1.0`, `mse_weight=0.05`, `lambda_order=0.5`, `lr=2e-4`, `patch_size=4`, `emb_dim=64`
- **Dataset-specific parameters:**
  - `FD001`: `seq_len=32`, `batch_size=32`, `epochs=24`, `weight_decay=1e-4`, `d_model=64`, `nhead=1`, `num_scales=2`, `ffn_dim=128`, `dropout=0.1`
  - `FD002`: `seq_len=64`, `batch_size=64`, `epochs=16`, `weight_decay=1e-2`, `d_model=128`, `nhead=4`, `num_scales=3`, `ffn_dim=256`, `dropout=0.2`
  - `FD003`: `seq_len=48`, `batch_size=32`, `epochs=20`, `weight_decay=1e-4`, `d_model=64`, `nhead=1`, `num_scales=2`, `ffn_dim=128`, `dropout=0.1`
  - `FD004`: `seq_len=64`, `batch_size=64`, `epochs=16`, `weight_decay=1e-2`, `d_model=128`, `nhead=4`, `num_scales=3`, `ffn_dim=256`, `dropout=0.2`

---

## Quick Start (Local, Server, or Kaggle)

### 1. Install dependencies & download data

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

*(In Kaggle, data is automatically discovered in `/kaggle/input/...` or `./data`.)*

### 2. Run a single experiment (`src/train.py` — Hydra & `--` CLI flags)

`src/train.py` supports both standard Hydra overrides (`key=value`) and `--` CLI flags (`--seed`, `--dataset`, `--exp`, `--epochs`, etc.):

```bash
# Default run (FD001, 00_full_model, seed=50)
python -m src.train

# Using '--' flags:
python -m src.train --dataset FD001 --seed 50 --exp 00_full_model
python -m src.train --dataset FD002 --seed 42 --exp 01_linear_head_standard_mse --epochs 16

# Using Hydra syntax:
python -m src.train data=fd003 seed=44 exp=05_wo_residual_proxy_head
```

### 3. Run the Full Multi-Seed Ablation Study (`run_ablation.py`)

```bash
# List all 8 experiments (00 - 07)
python run_ablation.py --list

# Run full model on all datasets for default seed=50
python run_ablation.py --exps 00_full_model --datasets all --seed 50

# Run specific experiments by number or prefix across all 5 seeds
python run_ablation.py --exps 0 1 2 3 4 5 6 7 --datasets all --seeds 42 43 44 45 50

# Or simply run the entire 8 x 4 x 5 matrix with automatic --resume:
python run_ablation.py --exps all --datasets all --seeds all

# Run leak-free validation (85% train / 15% val) to inspect validation curves:
python run_ablation.py --exps all --datasets all --seeds all --use_validation
```

### 4. Aggregate Results & Sync Kaggle Sessions

```bash
# Generate Markdown, CSV, and LaTeX (booktabs) summary tables
python aggregate_results.py --out_dir results

# Merge previous Kaggle session results and pack results_backup.zip
python sync_results.py --prev_results_dir /kaggle/input/exp2026-saved-results --out_dir /kaggle/working/results
```
