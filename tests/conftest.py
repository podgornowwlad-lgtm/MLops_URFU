"""Общие фикстуры: синтетический датасет со схемой Kaggle S4E1 и облегчённый конфиг.

Настоящие данные для тестов не нужны — CI не зависит от Kaggle.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bank_churn.config import ExperimentConfig, Settings, load_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def synthetic_churn(n_rows: int = 300, seed: int = 7) -> pd.DataFrame:
    """Данные с реалистичными диапазонами и зависимостью оттока от возраста и активности."""
    rng = np.random.default_rng(seed)
    age = rng.normal(38, 9, n_rows).clip(18, 92).round()
    active = rng.integers(0, 2, n_rows)
    products = rng.choice([1, 2, 3, 4], n_rows, p=[0.5, 0.4, 0.07, 0.03])
    has_money = rng.random(n_rows) < 0.45
    score = 0.07 * (age - 38) - 1.1 * active + 0.8 * (products == 1) - 1.2
    churn = rng.random(n_rows) < 1 / (1 + np.exp(-score))

    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "CustomerId": rng.integers(15_565_000, 15_816_000, n_rows),
            "Surname": rng.choice(["Hsueh", "Kao", "Chiemenam", "Okwudiliolisa"], n_rows),
            "CreditScore": rng.integers(350, 851, n_rows),
            "Geography": rng.choice(["France", "Germany", "Spain"], n_rows, p=[0.57, 0.21, 0.22]),
            "Gender": rng.choice(["Male", "Female"], n_rows),
            "Age": age,
            "Tenure": rng.integers(0, 11, n_rows),
            "Balance": np.where(has_money, rng.uniform(5e3, 2.5e5, n_rows).round(2), 0.0),
            "NumOfProducts": products,
            "HasCrCard": rng.integers(0, 2, n_rows).astype(float),
            "IsActiveMember": active.astype(float),
            "EstimatedSalary": rng.uniform(11, 2e5, n_rows).round(2),
            "Exited": churn.astype(int),
        }
    )


@pytest.fixture
def churn_df() -> pd.DataFrame:
    return synthetic_churn()


@pytest.fixture(scope="session")
def experiment() -> ExperimentConfig:
    """Боевой конфиг с маленькими сетками, чтобы тесты шли секунды."""
    raw = load_experiment(CONFIG_PATH).model_dump()
    raw["candidates"] = {
        "logreg": {"model__C": [0.1, 1.0]},
        "catboost": {"model__iterations": [20], "model__depth": [3]},
    }
    raw["tuning"].update(
        n_trials=2,
        timeout_sec=None,
        search_space={
            "model__iterations": {"kind": "int", "low": 10, "high": 30, "step": 10},
            "model__depth": {"kind": "int", "low": 2, "high": 4, "step": 2},
            "model__learning_rate": {"kind": "float", "low": 0.05, "high": 0.3, "log": True},
            "model__auto_class_weights": {"kind": "categorical", "choices": ["Balanced"]},
        },
    )
    return ExperimentConfig.model_validate(raw)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings без чтения .env разработчика; все артефакты — во временной папке."""
    data_path = tmp_path / "train.csv"
    synthetic_churn().to_csv(data_path, index=False)
    return Settings(
        _env_file=None,
        config_path=CONFIG_PATH,
        data_path=str(data_path),
        model_path=tmp_path / "models" / "model.joblib",
        reports_dir=tmp_path / "reports",
        n_jobs=1,
    )
