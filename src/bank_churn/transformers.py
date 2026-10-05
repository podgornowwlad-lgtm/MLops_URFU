"""Собственные sklearn-трансформеры для feature engineering из EDA.

В ноутбуке эти шаги были функциями в ``FunctionTransformer``. Классы с явными
параметрами корректно клонируются в ``GridSearchCV``, поддерживают
``get_feature_names_out`` и ``set_output(transform="pandas")``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from bank_churn.exceptions import DataValidationError


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Оставляет только признаки модели, в фиксированном порядке.

    Технические поля (``id``, ``CustomerId``, ``Surname``) и любые лишние колонки
    отбрасываются, а порядок колонок не зависит от входного CSV. Поэтому на
    инференс можно подавать сырые данные в исходном формате.
    """

    def __init__(self, columns: Sequence[str] = ()) -> None:
        self.columns = columns

    def fit(self, x: pd.DataFrame, y: object = None) -> FeatureSelector:
        self._check_columns(x)
        self.n_features_out_ = len(self.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        self._check_columns(x)
        return x.loc[:, list(self.columns)].copy()

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.columns, dtype=object)

    def _check_columns(self, x: pd.DataFrame) -> None:
        missing = [col for col in self.columns if col not in x.columns]
        if missing:
            raise DataValidationError(f"input is missing model features: {missing}")


class BalanceFlagger(TransformerMixin, BaseEstimator):
    """Добавляет бинарный признак «ненулевой баланс».

    По EDA ``Balance`` — zero-inflated признак: у большой доли клиентов баланс
    равен нулю, поэтому сам факт наличия денег на счёте несёт отдельный сигнал.
    """

    def __init__(self, source: str = "Balance", name: str = "HasBalance") -> None:
        self.source = source
        self.name = name

    def fit(self, x: pd.DataFrame, y: object = None) -> BalanceFlagger:
        if self.source not in x.columns:
            raise DataValidationError(f"column {self.source!r} is required for {self.name!r}")
        self.input_columns_ = list(x.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        result = x.copy()
        result[self.name] = (result[self.source] > 0).astype(int)
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        check_is_fitted(self)
        base = self.input_columns_ if input_features is None else list(input_features)
        return np.asarray([*base, self.name], dtype=object)
