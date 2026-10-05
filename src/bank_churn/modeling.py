"""Реестр моделей и сборка пайплайна «сырые данные → вероятность оттока»."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from bank_churn.config import ColumnsConfig
from bank_churn.transformers import BalanceFlagger, FeatureSelector

BALANCE_COLUMN = "Balance"


@dataclass(frozen=True)
class ModelSpec:
    """Как создать классификатор и нужна ли ему «линейная» предобработка."""

    factory: Callable[[int], Any]
    # Линейным моделям нужны импьютер, полиномиальные признаки и скейлинг,
    # деревьям и бустингу — нет (см. выводы EDA в ноутбуке).
    linear: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        lambda seed: LogisticRegression(max_iter=2000, random_state=seed), linear=True
    ),
    "decision_tree": ModelSpec(lambda seed: DecisionTreeClassifier(random_state=seed)),
    "random_forest": ModelSpec(lambda seed: RandomForestClassifier(random_state=seed)),
    "catboost": ModelSpec(
        lambda seed: CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False)
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}, available: {sorted(MODEL_REGISTRY)}") from None


def build_pipeline(
    model_name: str,
    columns: ColumnsConfig,
    *,
    seed: int = 42,
    exclude: Iterable[str] = (),
) -> Pipeline:
    """Собирает пайплайн для модели ``model_name``.

    Все преобразования (отбор признаков, feature engineering, кодирование) живут
    внутри пайплайна: сохранённая модель принимает данные в исходном формате.

    Args:
        model_name: ключ из :data:`MODEL_REGISTRY`.
        columns: роли колонок.
        seed: random state классификатора.
        exclude: признаки, которые нужно убрать из модели.
    """
    spec = get_model_spec(model_name)
    excluded = set(exclude)
    features = [col for col in columns.features if col not in excluded]
    numeric = [col for col in columns.numeric if col in features]
    categorical = [col for col in columns.categorical if col in features]

    encoders: list[tuple[str, Any, list[str]]] = [
        (
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            categorical,
        )
    ]
    if spec.linear:
        encoders.append(("numeric", SimpleImputer(strategy="median"), numeric))

    steps: list[tuple[str, Any]] = [("select", FeatureSelector(features))]
    if BALANCE_COLUMN in features:
        steps.append(("balance_flag", BalanceFlagger(source=BALANCE_COLUMN)))
    steps.append(
        (
            "encode",
            ColumnTransformer(encoders, remainder="passthrough", verbose_feature_names_out=False),
        )
    )
    if spec.linear:
        steps += [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    for _, transformer in steps:
        transformer.set_output(transform="pandas")

    steps.append(("model", spec.factory(seed)))
    return Pipeline(steps)
