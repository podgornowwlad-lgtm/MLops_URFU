from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from bank_churn.cli import app
from bank_churn.config import ExperimentConfig, Settings
from tests.conftest import synthetic_churn

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture
def cli_config(
    tmp_path: Path,
    settings: Settings,
    experiment: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Окружение как у пользователя: переменные CHURN_* и собственный YAML-конфиг."""
    monkeypatch.chdir(tmp_path)  # чтобы не подхватить .env разработчика
    monkeypatch.setenv("CHURN_DATA_PATH", settings.data_path)
    monkeypatch.setenv("CHURN_MODEL_PATH", str(settings.model_path))
    monkeypatch.setenv("CHURN_REPORTS_DIR", str(settings.reports_dir))
    monkeypatch.setenv("CHURN_N_JOBS", "1")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment.model_dump()), encoding="utf-8")
    return config_path


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("compare", "train", "predict"):
        assert command in result.output


def test_compare_command(cli_config: Path, settings: Settings) -> None:
    result = runner.invoke(app, ["compare", "-c", str(cli_config)])

    assert result.exit_code == 0, result.output
    assert (settings.reports_dir / "model_comparison.csv").exists()


def test_train_and_predict_commands(cli_config: Path, tmp_path: Path) -> None:
    trained = runner.invoke(app, ["train", "-c", str(cli_config), "--n-trials", "1"])
    assert trained.exit_code == 0, trained.output

    input_path = tmp_path / "new.csv"
    synthetic_churn(n_rows=10, seed=3).drop(columns="Exited").to_csv(input_path, index=False)
    output_path = tmp_path / "pred.csv"
    predicted = runner.invoke(
        app, ["predict", str(input_path), "-o", str(output_path), "--threshold", "0.3"]
    )

    assert predicted.exit_code == 0, predicted.output
    assert len(pd.read_csv(output_path)) == 10


def test_missing_config_fails_cleanly(cli_config: Path) -> None:
    result = runner.invoke(app, ["train", "-c", "no_such.yaml"])

    assert result.exit_code != 0
