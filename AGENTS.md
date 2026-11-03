# AGENTS.md

## What this is

Research code for Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan engine datasets (FD001–FD004) and a 5-seed Ablation Study (`42, 43, 44, 45, 50`).
Model: **STAREncoder + Spherical Proxy-Head + LDL (KL-Loss) + Cosine Gram Order + Decoupled Residual Head + Proxy-Centric 1/K Loss + Regime-Specific Normalization**.

## Stack

Python 3.10+ · PyTorch · Hydra / OmegaConf · scikit-learn · pandas · numpy · seaborn · matplotlib

Install: `pip install -r requirements.txt`
Download data: `python scripts/download_data.py`

## Entry points

| Command | Purpose |
|---------|---------|
| `python -m src.train` | Train a single configuration using Hydra (`data=fd001 seed=50`) or `--` CLI flags (`--dataset FD001 --seed 50 --exp 0`) |
| `python run_ablation.py` | Run ablation experiments (`00`–`07`) across datasets and seeds (`42, 43, 44, 45, 50`) with automatic `--resume` |
| `python sync_results.py` | Merge previous Kaggle session results with current runs and package `results_backup.zip` |
| `python aggregate_results.py` | Aggregate partial/complete results (`Mean ± Std`) into Markdown, CSV, and LaTeX (`booktabs`) |

### Examples

```bash
python run_ablation.py --list
python -m src.train --dataset FD001 --seed 50 --exp 00_full_model
python run_ablation.py --exps all --datasets all --seeds 42 43 44 45 50
python aggregate_results.py --out_dir results
```

## Strict Invariants

1. **Leak-Free Validation & Normalization (`src/data.py`)**:
   - When `use_validation=True`, engine units are split into `train_units` and `val_units` **before** calling `apply_regime_specific_normalization`, so `KMeans.fit` and cluster `mins`/`maxs` are computed strictly on `real_train_df` and applied to `val_df` and `test_df`.
   - `extract_windows` always sorts by `time` (`.sort_values('time')`) and left-pads trajectories shorter than `seq_len` using `np.pad(data_matrix, ((pad_len, 0), (0, 0)), mode='edge')` so no short test engines are ever dropped.
2. **RNG Seed (`seed: 50`) in `src/models.py`**:
   - Preserves exact layer creation order in `ProxySREncoder.__init__` (`self.proxies` initialized before `self.res_head`) and decoupled parameter groups in `build_optimizer` (`no_decay` for `proxies`, `pos_embed`, `pos_bias`, `norm`, and `.bias`).
