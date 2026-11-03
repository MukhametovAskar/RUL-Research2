from copy import deepcopy
from typing import Any, Dict, List

from .base_config import get_dataset_config

ABLATION_GRID: Dict[str, Dict[str, Any]] = {
    "00_full_model": {
        "block": "A. Full Model",
        "title": "Full Proposed Model",
        "latex_name": r"\textbf{Full Proposed Model (Ours)}",
        "description": (
            "STAREncoder + Spherical Proxy-Head + LDL (KL-Loss) + Cosine Gram Order "
            "+ Decoupled Residual Head + Proxy-Centric 1/K Loss + Regime-Specific Norm (FD002/FD004)."
        ),
        "overrides": {},
    },
    "01_linear_head_standard_mse": {
        "block": "B. Baselines",
        "title": "Linear Head + Standard MSE (1/B)",
        "latex_name": r"Linear Head + Standard MSE ($1/B$)",
        "description": (
            "Замена Proxy-Head и Residual-Head на прямой линейный регрессор Linear -> RUL "
            "с обучением на стандартном MSE по батчу (1/B)."
        ),
        "overrides": {
            "head_type": "linear",
            "use_residual": False,
            "use_proxy_1k": False,
            "mse_weight": 1.0,
            "kl_weight": 0.0,
            "lambda_order": 0.0,
        },
    },
    "02_linear_head_balanced_mse_1k": {
        "block": "B. Baselines",
        "title": "Linear Head + Balanced MSE (1/K)",
        "latex_name": r"Linear Head + Balanced MSE ($1/K$)",
        "description": (
            "Прямой линейный регрессор Linear -> RUL без прокси, обучаемый со сбалансированным "
            "по сетке 1/K MSE-лоссом."
        ),
        "overrides": {
            "head_type": "linear",
            "use_residual": False,
            "use_proxy_1k": True,
            "mse_weight": 1.0,
            "kl_weight": 0.0,
            "lambda_order": 0.0,
        },
    },
    "03_wo_ldl_kl_loss": {
        "block": "C. Component Ablations",
        "title": "w/o LDL / KL-Loss (kl_weight = 0)",
        "latex_name": r"w/o LDL / KL-Loss ($\lambda_{\text{KL}} = 0$)",
        "description": (
            "Оставляем Proxy-Head и Residual-Head, но выключаем KL-Loss (kl_weight = 0), "
            "обучая модель только по MSE и Gram-регуляризатору."
        ),
        "overrides": {
            "kl_weight": 0.0,
        },
    },
    "04_wo_proxy_1k_aggregation": {
        "block": "C. Component Ablations",
        "title": "w/o Proxy-Centric 1/K Aggregation (Standard 1/B)",
        "latex_name": r"w/o Proxy-Centric $1/K$ Aggregation (Standard $1/B$)",
        "description": (
            "Оставляем всю архитектуру прокси, но возвращаем обычное усреднение лосса "
            "по батчу (1/B) вместо усреднения по активным прокси (1/K)."
        ),
        "overrides": {
            "use_proxy_1k": False,
        },
    },
    "05_wo_residual_proxy_head": {
        "block": "C. Component Ablations",
        "title": "w/o Residual Proxy Head (Coarse Proxies Only)",
        "latex_name": r"w/o Residual Proxy Head (Coarse Proxies Only)",
        "description": (
            "Отключаем локальную остаточную поправку delta_rul (pred_rul = proxy_rul)."
        ),
        "overrides": {
            "use_residual": False,
        },
    },
    "06_wo_cosine_gram_regularizer": {
        "block": "C. Component Ablations",
        "title": "w/o Cosine Gram Regularizer (lambda_order = 0)",
        "latex_name": r"w/o Cosine Gram Regularizer ($\lambda_{\text{order}} = 0$)",
        "description": (
            "Отключаем косинусный регуляризатор матрицы Грама (lambda_order = 0)."
        ),
        "overrides": {
            "lambda_order": 0.0,
        },
    },
    "07_wo_regime_normalization": {
        "block": "C. Component Ablations",
        "title": "w/o Regime-Specific Normalization (Global Norm on FD002/FD004)",
        "latex_name": r"w/o Regime-Specific Norm (Global Norm on FD002/FD004)",
        "description": (
            "Заменяем K-Means (6 кластеров) нормализацию по рабочим режимам на глобальную "
            "нормализацию (1 кластер). Запускается на FD002/FD004 (на FD001/FD003 дублирует 00_full_model)."
        ),
        "overrides": {
            "norm_mode": "global",
        },
        "only_datasets": ["FD002", "FD004"],
        "duplicate_from": "00_full_model",
    },
}

ALL_PRESETS: Dict[str, Dict[str, Any]] = dict(ABLATION_GRID)


def list_ablation_names() -> List[str]:
    return list(ABLATION_GRID.keys())


def resolve_experiment_names(exp_queries: List[str]) -> List[str]:
    if not exp_queries or any(str(q).lower() == "all" for q in exp_queries):
        return list(ABLATION_GRID.keys())

    resolved: List[str] = []
    for query in exp_queries:
        q = str(query).strip()
        if q in ALL_PRESETS:
            if q not in resolved:
                resolved.append(q)
            continue

        # Поддержка цифр '0', '1' .. '7' и '00', '01' .. '07'
        q_padded = f"0{q}" if len(q) == 1 and q.isdigit() else q
        matches = [
            name
            for name in ALL_PRESETS.keys()
            if name.startswith(q) or name.startswith(f"{q_padded}_") or name == q_padded
        ]
        if matches:
            for m in matches:
                if m not in resolved:
                    resolved.append(m)
        else:
            valid = ", ".join(ABLATION_GRID.keys())
            raise ValueError(f"Unknown experiment name or prefix '{q}'. Valid experiments: {valid}")

    return resolved


def get_ablation_config(exp_name: str, dataset_name: str, seed: int = 50) -> Dict[str, Any]:
    if exp_name not in ALL_PRESETS:
        matches = resolve_experiment_names([exp_name])
        exp_name = matches[0]

    preset = ALL_PRESETS[exp_name]
    cfg = get_dataset_config(dataset_name)
    cfg["seed"] = int(seed)
    cfg["exp_name"] = exp_name
    cfg["exp_block"] = preset["block"]
    cfg["exp_title"] = preset["title"]
    cfg["exp_latex_name"] = preset["latex_name"]
    cfg["exp_description"] = preset["description"]

    overrides = deepcopy(preset.get("overrides", {}))
    cfg.update(overrides)
    return cfg
