"""Метрики качества и интерпретация модели."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def to_labels(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Бинарный прогноз по порогу вероятности."""
    return (np.asarray(proba) >= threshold).astype(int)


def compute_metrics(
    y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Метрики классификации на отложенной выборке."""
    y_pred = to_labels(proba, threshold)
    return {
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def text_report(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> str:
    """``classification_report`` в текстовом виде — для артефакта отчёта."""
    report: str = classification_report(y_true, to_labels(proba, threshold), digits=3)
    return report


def feature_importances(pipeline: Pipeline) -> pd.Series | None:
    """Важности признаков финальной модели (если модель их предоставляет)."""
    model = pipeline[-1]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline[:-1].get_feature_names_out()
    return (
        pd.Series(model.feature_importances_, index=names, name="importance")
        .sort_values(ascending=False)
        .rename_axis("feature")
    )
