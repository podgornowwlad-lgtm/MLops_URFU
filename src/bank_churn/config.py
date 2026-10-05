"""Конфигурация проекта.

Настройки разделены на два уровня:

* :class:`Settings` — параметры окружения: пути к данным и артефактам, число потоков,
  уровень логирования. Читаются из переменных окружения с префиксом ``CHURN_`` и из
  файла ``.env`` (в git не коммитится, шаблон — ``.env.example``).
* :class:`ExperimentConfig` — параметры эксперимента: колонки, сетки гиперпараметров,
  пространство поиска Optuna. Хранятся в YAML рядом с кодом и версионируются в git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Metric = Literal["recall", "precision", "f1", "roc_auc"]


class Settings(BaseSettings):
    """Параметры окружения (переменные ``CHURN_*`` или файл ``.env``)."""

    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    config_path: Path = Path("configs/train.yaml")
    # str, а не Path: pandas умеет читать и локальный путь, и URL.
    data_path: str = "data/raw/train.csv"
    model_path: Path = Path("models/churn_model.joblib")
    reports_dir: Path = Path("reports")
    n_jobs: int = -1
    log_level: str = "INFO"


class _Frozen(BaseModel):
    """База для секций YAML: опечатки в ключах — ошибка, а не тихое игнорирование."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ColumnsConfig(_Frozen):
    """Роли колонок датасета (по итогам EDA)."""

    target: str
    ids: list[str] = Field(description="Технические идентификаторы и ПДн — в модель не идут")
    numeric: list[str]
    binary: list[str] = []
    categorical: list[str] = []

    @property
    def features(self) -> list[str]:
        """Все признаки модели в фиксированном порядке."""
        return [*self.numeric, *self.binary, *self.categorical]

    @model_validator(mode="after")
    def _check_groups_do_not_overlap(self) -> ColumnsConfig:
        listed = [self.target, *self.ids, *self.features]
        duplicated = sorted({col for col in listed if listed.count(col) > 1})
        if duplicated:
            raise ValueError(f"columns are listed in several groups: {duplicated}")
        return self


class ValidationConfig(_Frozen):
    """Схема валидации: отложенная выборка + стратифицированная кросс-валидация."""

    test_size: float = Field(0.2, gt=0, lt=1)
    split_seed: int = 42
    n_splits: int = Field(3, ge=2)
    cv_seed: int = 52
    metric: Metric = "recall"


class IntParam(_Frozen):
    kind: Literal["int"]
    low: int
    high: int
    step: int = 1
    log: bool = False


class FloatParam(_Frozen):
    kind: Literal["float"]
    low: float
    high: float
    log: bool = False


class CategoricalParam(_Frozen):
    kind: Literal["categorical"]
    choices: list[Any] = Field(min_length=1)


ParamSpec = Annotated[IntParam | FloatParam | CategoricalParam, Field(discriminator="kind")]


class TuningConfig(_Frozen):
    """Подбор гиперпараметров финальной модели с помощью Optuna."""

    model: str
    excluded_features: list[str] = Field(
        default_factory=list,
        description="Малоинформативные признаки (по SHAP / feature importance)",
    )
    n_trials: int = Field(20, ge=1)
    timeout_sec: int | None = Field(None, ge=1)
    threshold: float = Field(0.5, gt=0, lt=1)
    search_space: dict[str, ParamSpec]


class ExperimentConfig(_Frozen):
    """Полное описание эксперимента из ``configs/*.yaml``."""

    seed: int = 42
    columns: ColumnsConfig
    validation: ValidationConfig = ValidationConfig()
    # Ключи сеток — штатные имена параметров sklearn-пайплайна (`model__C`, `poly__degree`).
    candidates: dict[str, dict[str, list[Any]]]
    tuning: TuningConfig

    @model_validator(mode="after")
    def _check_excluded_features(self) -> ExperimentConfig:
        unknown = sorted(set(self.tuning.excluded_features) - set(self.columns.features))
        if unknown:
            raise ValueError(f"tuning.excluded_features are not model features: {unknown}")
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Читает и валидирует YAML-конфиг эксперимента."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)
