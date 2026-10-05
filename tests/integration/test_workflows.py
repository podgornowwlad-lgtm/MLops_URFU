import json

import pandas as pd
import pytest

from bank_churn.config import ExperimentConfig, Settings
from bank_churn.workflows import (
    COMPARISON_FILE,
    IMPORTANCES_FILE,
    METRICS_FILE,
    REPORT_FILE,
    run_comparison,
    run_prediction,
    run_training,
)
from tests.conftest import synthetic_churn

pytestmark = pytest.mark.integration


def test_comparison_ranks_candidates(settings: Settings, experiment: ExperimentConfig) -> None:
    table = run_comparison(settings, experiment)

    assert set(table["model"]) == set(experiment.candidates)
    assert table["cv_recall"].is_monotonic_decreasing
    assert (settings.reports_dir / COMPARISON_FILE).exists()


def test_training_produces_all_artifacts(settings: Settings, experiment: ExperimentConfig) -> None:
    metadata = run_training(settings, experiment)

    assert settings.model_path.exists()
    for name in (METRICS_FILE, REPORT_FILE, IMPORTANCES_FILE):
        assert (settings.reports_dir / name).exists(), name
    saved = json.loads((settings.reports_dir / METRICS_FILE).read_text(encoding="utf-8"))
    assert saved["test_metrics"] == pytest.approx(metadata["test_metrics"])
    assert not set(saved["features"]) & set(experiment.tuning.excluded_features)
    assert saved["n_trials"] == experiment.tuning.n_trials


def test_train_then_predict_new_clients(settings: Settings, experiment: ExperimentConfig) -> None:
    run_training(settings, experiment)
    new_clients = synthetic_churn(n_rows=25, seed=99).drop(columns="Exited")
    input_path = settings.reports_dir / "new.csv"
    new_clients.to_csv(input_path, index=False)
    output_path = settings.reports_dir / "out" / "pred.csv"

    run_prediction(settings, input_path, output_path)

    saved = pd.read_csv(output_path)
    assert len(saved) == 25
    assert saved["id"].tolist() == new_clients["id"].tolist()
