# STAR-Proxy RUL — Spatio-Temporal Attention & Spherical Proxy Metric Learning for Turbofan RUL Prediction

Official research implementation of **STAR-Proxy RUL** on the NASA C-MAPSS benchmark datasets (`FD001`, `FD002`, `FD003`, `FD004`), including a complete 2-Stage 5-Seed Ablation Study (`Seeds: 42, 43, 44, 45, 50`).

---

## Key Contributions & Architecture

1. **Backbone (`STAREncoder` in `src/backbone.py`)**:
   - Dimension-wise temporal patch embedding (`PatchEmbedDimWise`) with learnable positional encoding.
   - Hierarchical spatio-temporal attention blocks (`STARAttentionBlock`: Intra-Sensor Temporal MHA + Inter-Sensor Spatial MHA).
   - Multi-scale temporal patch merging (`PatchMerging`) and positional attention pooling (`PatchAttnPool`).
2. **Spherical Proxy-Head & Decoupled Residual Refinement (`ProxySREncoder` in `src/models.py`)**:
   - $L_2$-normalized hyperspherical embedding space with $K=16$ orthogonal learnable proxy attractors spanning the extended RUL grid $[-10, 135]$ ($\text{step} \approx 9.67$ cycles) at sharp temperature $\tau = 0.02$.
   - Coarse-to-fine decomposition: coarse spherical proxy expectation $\hat{y}_{\text{proxy}} = \sum_{k=1}^K w_k r_k$ followed by a bounded local residual correction $\Delta\hat{y} \in [-\Delta, +\Delta]$ via `proxy_rul.detach() + delta_rul`, preventing regression gradients from distorting hyperspherical proxy geometry.
3. **Hybrid Proxy Objective (`HybridProxyLoss` in `src/losses.py`)**:
   - **Proxy-Centric $1/K$ Loss Aggregation**: Normalizes sample losses within each active proxy attractor (`proxy_mass > 0.35`) and averages equally across active proxies ($1/K$), preventing the healthy constant RUL plateau ($\text{RUL}=125$) from dominating critical transition/degradation regimes.
   - **Label Distribution Learning (LDL)**: Soft Gaussian target distributions ($\sigma = 0.5 \cdot \text{step}$) supervised via KL-divergence ($\lambda_{\text{KL}} = 1.0$).
   - **Cosine Gram Order Regularizer**: Enforces smooth geodesic ordering of proxies along a great-circle arc ($\cos(\pi \cdot |r_i - r_j| / (r_{\max} - r_{\min}))$, $\lambda_{\text{order}} = 0.5$).
4. **Leak-Free Stratified Validation & Regime-Specific Normalization (`src/data.py`)**:
   - Operating regimes on multi-condition datasets (`FD002`, `FD004`) are clustered via `KMeans(n_clusters=6)` and normalized per cluster.
   - In validation mode (`use_validation=true`), engine units are split into stratified `train_units` (85%) and `val_units` (15%) across lifespan quantiles **strictly before** fitting `KMeans` and computing cluster `min`/`max` statistics.

---

## Repository Structure

```text
RUL-Research2/
├── configs/                        # Hydra YAML configuration hierarchy & ablation presets
│   ├── config.yaml                 # Root Hydra config
│   ├── base_config.py              # Dataset & seed defaults, deterministic seeding (set_seed)
│   ├── ablation_grid.py            # Canonical definition of Ablation Experiments 00..07
│   ├── data/                       # Dataset-specific configs (fd001.yaml .. fd004.yaml)
│   ├── model/                      # Model architecture configs (proxy.yaml, linear.yaml)
│   ├── trainer/                    # Optimizer, scheduler, loss & validation configs (default.yaml)
│   └── experiment/                 # Experiment overrides (00_full_model.yaml .. 07_wo_regime_normalization.yaml)
├── src/                            # Core library modules
│   ├── data.py                     # Leak-free stratified split, KMeans regime normalization, windowing
│   ├── backbone.py                 # PatchEmbedDimWise, STARAttentionBlock, PatchMerging, STAREncoder
│   ├── models.py                   # ProxySREncoder, PatchAttnPool, decoupled AdamW param groups
│   ├── losses.py                   # HybridProxyLoss (LDL KL, Proxy-Centric 1/K, Cosine Gram Order)
│   ├── trainer.py                  # 2-Stage training/evaluation loop, EMA, checkpointing, ledgers
│   ├── visualization.py            # Zonal RMSE [0-20, 20-50, 50-100, 100-150], SMART PCCs, plots
│   └── train.py                    # Hydra + CLI unified entrypoint
├── scripts/                        # Helper & verification scripts
│   ├── train.py                    # CLI wrapper for Hydra experiment execution
│   ├── download_data.py            # Automatic NASA C-MAPSS dataset downloader
│   └── verify_ablation.py          # Protocol & table audit script (verifies consensus E* & zero leakage)
├── notebooks/                      # Exploratory & analysis notebooks
├── data/                           # Local directory for raw C-MAPSS text files (ignored by Git)
├── run_ablation.py                 # Automated multi-seed ablation runner with --resume support
├── aggregate_results.py            # Generates Markdown, CSV, and LaTeX (booktabs) reports
├── sync_results.py                 # Merges multi-session runs and packages results_backup.zip
├── ABLATION_STUDY_RESULTS.md       # Complete 5-seed benchmark & ablation tables (Stages 1 & 2)
└── requirements.txt                # Pinned Python package dependencies
```

---

## Benchmark & Ablation Study Results (5 Seeds: `42, 43, 44, 45, 50`)

### 2-Stage Consensus Validation Protocol
- **Stage 1 (`85% Train / 15% Val`, 5-split Monte-Carlo CV)**: Five stratified `85/15` splits are trained using seeds `42, 43, 44, 45, 50`. To eliminate single-split variance on small 15% validation subsets without any test leakage, the stopping epoch $E^*$ for each `(experiment, dataset)` pair is selected via the **consensus median** across the 5 validation splits:
  $$E^* = \operatorname{median}_{s \in \{42, 43, 44, 45, 50\}}\!\left(E_{\text{val\_min}}^{(s)}\right)$$
- **Stage 2 (`100% Train -> Test`)**: All 5 seeds are trained on 100% of the training engines, and test metrics (`Test_RMSE`, `Test_S_Score`, and zonal `RMSE`) are evaluated strictly at the consensus epoch $E^*$.

| # | Experiment | FD001 RMSE | FD001 S-Score | FD002 RMSE | FD002 S-Score | FD003 RMSE | FD003 S-Score | FD004 RMSE | FD004 S-Score | Macro RMSE | Macro S-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `00_full_model` (**Full Proposed Model**) | **11.09 ± 0.23** | **165.48 ± 10.08** | **12.82 ± 0.46** | **632.43 ± 31.44** | **11.82 ± 0.38** | **346.13 ± 49.15** | **14.07 ± 0.59** | **857.63 ± 65.98** | **12.45 ± 0.42** | **500.42 ± 39.16** |
| **1** | `01_linear_mse_1b` (Linear Head + Standard MSE $1/B$) | 11.95 ± 0.45 | 219.34 ± 16.31 | 13.67 ± 1.19 | 854.74 ± 149.66 | 11.98 ± 0.64 | 325.00 ± 53.92 | 14.16 ± 0.95 | 996.75 ± 122.55 | 12.94 ± 0.81 | 598.96 ± 85.61 |
| **2** | `02_linear_balanced_1k` (Linear Head + Balanced MSE $1/K$) | 11.23 ± 0.59 | 183.69 ± 20.58 | 14.29 ± 1.19 | 907.81 ± 201.38 | 11.11 ± 0.36 | 215.20 ± 22.37 | 14.99 ± 0.95 | 926.26 ± 125.77 | 12.90 ± 0.77 | 558.24 ± 92.53 |
| **3** | `03_wo_ldl_kl` (w/o LDL / KL-Loss, $\lambda_{\text{KL}}=0$) | 11.13 ± 0.33 | 166.55 ± 9.73 | 13.37 ± 0.29 | 727.50 ± 35.21 | 11.44 ± 0.55 | 261.64 ± 44.86 | 15.48 ± 1.47 | 981.82 ± 113.27 | 12.85 ± 0.66 | 534.38 ± 50.77 |
| **4** | `04_wo_proxy_1k` (w/o Proxy-Centric $1/K$ Aggregation) | 11.46 ± 0.24 | 189.81 ± 11.67 | 12.73 ± 0.22 | 685.61 ± 45.20 | 12.36 ± 0.45 | 487.26 ± 170.28 | 13.93 ± 0.50 | 990.98 ± 105.47 | 12.62 ± 0.35 | 588.41 ± 83.16 |
| **5** | `05_wo_residual_head` (w/o Decoupled Residual Head) | 11.42 ± 0.28 | 173.02 ± 10.05 | 13.54 ± 0.79 | 850.65 ± 290.61 | 11.69 ± 0.86 | 371.19 ± 235.56 | 14.11 ± 0.30 | 900.22 ± 53.63 | 12.69 ± 0.56 | 573.77 ± 147.46 |
| **6** | `06_wo_gram_order` (w/o Cosine Gram Regularizer) | 11.24 ± 0.32 | 171.59 ± 13.45 | 12.97 ± 0.55 | 667.46 ± 60.34 | 11.96 ± 0.15 | 383.37 ± 86.34 | 14.31 ± 0.38 | 880.87 ± 52.03 | 12.62 ± 0.35 | 525.82 ± 53.04 |
| **7** | `07_wo_regime_normalization` (w/o Regime Normalization) | 11.09 ± 0.23 | 165.48 ± 10.08 | 19.41 ± 1.20 | 5462.84 ± 4753.75 | 11.82 ± 0.38 | 346.13 ± 49.15 | 26.32 ± 1.20 | 32280.99 ± 24984.26 | 17.16 ± 0.76 | 9563.86 ± 7449.31 |

Detailed per-seed validation & test logs and zonal decompositions are available in [ABLATION_STUDY_RESULTS.md](ABLATION_STUDY_RESULTS.md).

---

## Default Hyperparameters

- **Evaluation Seeds (5 seeds):** `42, 43, 44, 45, 50`
- **Model & Loss (`configs/model/proxy.yaml`, `configs/trainer/default.yaml`):**
  - `max_rul=125`, `num_proxies=16`, `proxy_margin=10.0` (grid $[-10, 135]$), `temp=0.02`, `emb_dim=64`, `patch_size=4`, `n_layers_per_scale=2`
  - `kl_weight=1.0`, `mse_weight=0.05`, `lambda_order=0.5`, `ldl_sigma_ratio=0.5`, `proxy_active_threshold=0.35` (`proxy_mass > 0.35`)
  - `ema_decay=0.99`, `lr=2e-4`, `eta_min=1e-5`, `grad_clip_norm=1.0`
- **Dataset-Specific (`configs/data/fd001.yaml` – `fd004.yaml`):**
  - `FD001` (1 regime): `window_size=32`, `batch_size=32`, `epochs=30`, `t_max=30`, `weight_decay=1e-4`, `d_model=64`, `nhead=1`, `num_scales=2`, `ffn_dim=128`, `dropout=0.1`
  - `FD002` (6 regimes): `window_size=64`, `batch_size=64`, `epochs=16`, `t_max=16`, `weight_decay=1e-2`, `d_model=128`, `nhead=4`, `num_scales=3`, `ffn_dim=256`, `dropout=0.2`
  - `FD003` (1 regime): `window_size=48`, `batch_size=32`, `epochs=30`, `t_max=30`, `weight_decay=1e-4`, `d_model=64`, `nhead=1`, `num_scales=2`, `ffn_dim=128`, `dropout=0.1`
  - `FD004` (6 regimes): `window_size=64`, `batch_size=64`, `epochs=16`, `t_max=16`, `weight_decay=1e-2`, `d_model=128`, `nhead=4`, `num_scales=3`, `ffn_dim=256`, `dropout=0.2`

---

## How to Reproduce

### 1. Environment Setup & Data Preparation

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

### 2. Single Experiment Execution via Hydra CLI

Any experiment (`00`–`07`), dataset (`fd001`–`fd004`), seed, or hyperparameter can be overridden directly from the terminal via `scripts/train.py` or `python -m src.train`:

```bash
# Run the full proposed model on FD001 (seed 42):
python scripts/train.py experiment=00_full_model data=fd001 seed=42

# Run ablation experiment 03 (w/o LDL / KL-Loss) via short or canonical name:
python scripts/train.py experiment=03_wo_ldl_kl data=fd002 seed=42
python -m src.train experiment=03_wo_ldl_kl_loss data=fd002 seed=42

# Stage 1 (85% Train / 15% Stratified Val) for a specific experiment:
python scripts/train.py experiment=04_wo_proxy_1k data=fd003 seed=42 trainer.use_validation=true

# Stage 2 (100% Train -> Test) evaluated at consensus epoch E*:
python scripts/train.py experiment=04_wo_proxy_1k data=fd003 seed=42 trainer.use_validation=false trainer.target_epoch=18
```

### 3. Full Multi-Seed Ablation Pipeline (`run_ablation.py`)

```bash
# List all 8 ablation experiments:
python run_ablation.py --list

# Stage 1: Run 5-seed stratified validation (85/15) across all experiments and datasets:
python run_ablation.py --exps all --datasets all --seeds 42 43 44 45 50 --use_validation

# Stage 2: Run 5-seed full training (100% train -> test) with automatic resume:
python run_ablation.py --exps all --datasets all --seeds 42 43 44 45 50
```

### 4. Aggregation & Protocol Verification

```bash
# Aggregate all runs into Markdown, CSV, and LaTeX (booktabs) using consensus median epoch E*:
python aggregate_results.py --out_dir results --mode median_raw

# Audit all 320 runs (160 validation + 160 test) and verify ABLATION_STUDY_RESULTS.md:
python scripts/verify_ablation.py
```
