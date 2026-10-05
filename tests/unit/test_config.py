from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bank_churn.config import (
    CategoricalParam,
    ExperimentConfig,
    IntParam,
    Settings,
    load_experiment,
)
from tests.conftest import CONFIG_PATH


def _raw_config() -> dict[str, Any]:
    return load_experiment(CONFIG_PATH).model_dump()


def test_project_config_is_valid() -> None:
    cfg = load_experiment(CONFIG_PATH)

    assert cfg.columns.target == "Exited"
    assert cfg.validation.metric == "recall"
    assert cfg.tuning.model in cfg.candidates
    assert isinstance(cfg.tuning.search_space["model__depth"], IntParam)
    assert isinstance(cfg.tuning.search_space["model__auto_class_weights"], CategoricalParam)


def test_features_exclude_target_and_ids() -> None:
    columns = load_experiment(CONFIG_PATH).columns

    assert columns.target not in columns.features
    assert not set(columns.ids) & set(columns.features)


def test_unknown_key_is_rejected() -> None:
    raw = _raw_config()
    raw["validation"]["n_split"] = 5  # опечатка в ключе

    with pytest.raises(ValidationError, match="n_split"):
        ExperimentConfig.model_validate(raw)


def test_overlapping_column_groups_are_rejected() -> None:
    raw = _raw_config()
    raw["columns"]["categorical"].append("Age")

    with pytest.raises(ValidationError, match="several groups"):
        ExperimentConfig.model_validate(raw)


def test_excluded_features_must_exist() -> None:
    raw = _raw_config()
    raw["tuning"]["excluded_features"] = ["NoSuchColumn"]

    with pytest.raises(ValidationError, match="NoSuchColumn"):
        ExperimentConfig.model_validate(raw)


def test_search_space_requires_bounds() -> None:
    raw = _raw_config()
    raw["tuning"]["search_space"] = {"model__depth": {"kind": "int", "low": 2}}

    with pytest.raises(ValidationError, match="high"):
        ExperimentConfig.model_validate(raw)


def test_settings_are_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHURN_N_JOBS", "4")
    monkeypatch.setenv("CHURN_MODEL_PATH", "artifacts/m.joblib")

    settings = Settings(_env_file=None)

    assert settings.n_jobs == 4
    assert settings.model_path == Path("artifacts/m.joblib")
