"""Инференс: скоринг новых клиентов сохранённой моделью."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bank_churn.artifacts import ModelBundle, load_bundle
from bank_churn.evaluation import to_labels


class ChurnPredictor:
    """Обёртка над сохранённой моделью для прогноза по сырым данным."""

    def __init__(self, bundle: ModelBundle, id_column: str = "id") -> None:
        self._bundle = bundle
        self.id_column = id_column

    @classmethod
    def from_file(cls, path: str | Path, id_column: str = "id") -> ChurnPredictor:
        return cls(load_bundle(path), id_column=id_column)

    @property
    def threshold(self) -> float:
        return self._bundle.threshold

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Вероятность оттока для каждой строки."""
        return np.asarray(self._bundle.pipeline.predict_proba(df)[:, 1], dtype=float)

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """Таблица ``[id], churn_probability, churn_flag`` в порядке входных строк."""
        proba = self.predict_proba(df)
        result = pd.DataFrame(
            {
                "churn_probability": proba,
                "churn_flag": to_labels(proba, self.threshold if threshold is None else threshold),
            },
            index=df.index,
        )
        if self.id_column in df.columns:
            result.insert(0, self.id_column, df[self.id_column].to_numpy())
        return result.reset_index(drop=True)
