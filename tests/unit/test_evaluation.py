import numpy as np
import pandas as pd
import pytest

from bank_churn.config import ExperimentConfig
from bank_churn.evaluation import compute_metrics, feature_importances, text_report, to_labels
from bank_churn.modeling import build_pipeline


def test_threshold_controls_labels() -> None:
    proba = np.array([0.2, 0.5, 0.7])

    assert to_labels(proba, 0.5).tolist() == [0, 1, 1]
    assert to_labels(proba, 0.6).tolist() == [0, 0, 1]


def test_metrics_on_perfect_predictions() -> None:
    y = pd.Series([0, 1, 1, 0])
    proba = np.array([0.1, 0.9, 0.8, 0.3])

    metrics = compute_metrics(y, proba)

    assert metrics == pytest.approx(
        {"recall": 1.0, "precision": 1.0, "f1": 1.0, "accuracy": 1.0, "roc_auc": 1.0}
    )
    assert "recall" in text_report(y, proba)


def test_feature_importances_are_named(
    churn_df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    tree = build_pipeline("decision_tree", experiment.columns)
    tree.fit(churn_df.drop(columns="Exited"), churn_df["Exited"])

    importances = feature_importances(tree)

    assert importances is not None
    assert "Age" in importances.index
    assert importances.sum() == pytest.approx(1.0)


def test_feature_importances_absent_for_linear_model(
    churn_df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    logreg = build_pipeline("logreg", experiment.columns)
    logreg.fit(churn_df.drop(columns="Exited"), churn_df["Exited"])

    assert feature_importances(logreg) is None
