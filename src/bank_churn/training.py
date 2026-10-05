"""Отбор моделей (GridSearchCV) и подбор гиперпараметров финальной модели (Optuna)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from bank_churn.config import (
    CategoricalParam,
    ExperimentConfig,
    FloatParam,
    IntParam,
    ParamSpec,
    ValidationConfig,
)
from bank_churn.modeling import build_pipeline

logger = logging.getLogger(__name__)

# precision с zero_division=0: на маленьких фолдах модель может не предсказать ни одного «1».
SCORERS: dict[str, Any] = {
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
    "roc_auc": "roc_auc",
}


@dataclass(frozen=True)
class TuningResult:
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_score: float
    n_trials: int


def make_cv(validation: ValidationConfig) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=validation.n_splits, shuffle=True, random_state=validation.cv_seed
    )


def compare_candidates(
    x: pd.DataFrame, y: pd.Series, cfg: ExperimentConfig, *, n_jobs: int = -1
) -> pd.DataFrame:
    """Подбирает гиперпараметры каждого кандидата по сетке и сводит результаты в таблицу.

    В ноутбуке это были четыре скопированных блока ячеек; здесь — один цикл.
    Для контроля переобучения рядом с CV-метрикой выводится метрика на train.
    """
    metric = cfg.validation.metric
    rows: list[dict[str, Any]] = []

    for name, grid in cfg.candidates.items():
        logger.info("Grid search: %s", name)
        search = GridSearchCV(
            build_pipeline(name, cfg.columns, seed=cfg.seed),
            grid,
            scoring=SCORERS,
            refit=metric,
            cv=make_cv(cfg.validation),
            n_jobs=n_jobs,
            return_train_score=True,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(x, y)
        elapsed = time.perf_counter() - started

        best = search.best_index_
        results = search.cv_results_
        row: dict[str, Any] = {"model": name}
        row.update({f"cv_{m}": float(results[f"mean_test_{m}"][best]) for m in SCORERS})
        row[f"train_{metric}"] = float(results[f"mean_train_{metric}"][best])
        row["overfit_gap"] = row[f"train_{metric}"] - row[f"cv_{metric}"]
        row["refit_time_sec"] = float(search.refit_time_)
        row["search_time_sec"] = elapsed
        row["best_params"] = json.dumps(search.best_params_, default=str)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(f"cv_{metric}", ascending=False, ignore_index=True)


def suggest_params(trial: optuna.Trial, space: Mapping[str, ParamSpec]) -> dict[str, Any]:
    """Сэмплирует гиперпараметры из пространства поиска, описанного в конфиге."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, IntParam):
            params[name] = trial.suggest_int(
                name, spec.low, spec.high, step=spec.step, log=spec.log
            )
        elif isinstance(spec, FloatParam):
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, CategoricalParam):
            params[name] = trial.suggest_categorical(name, spec.choices)
    return params


def tune_final_model(
    x: pd.DataFrame, y: pd.Series, cfg: ExperimentConfig, *, n_jobs: int = -1
) -> TuningResult:
    """Optuna (TPE) максимизирует CV-метрику; лучший пайплайн дообучается на всём train."""
    tuning = cfg.tuning
    metric = cfg.validation.metric
    cv = make_cv(cfg.validation)

    def build() -> Pipeline:
        return build_pipeline(
            tuning.model, cfg.columns, seed=cfg.seed, exclude=tuning.excluded_features
        )

    def objective(trial: optuna.Trial) -> float:
        pipeline = build().set_params(**suggest_params(trial, tuning.search_space))
        scores = cross_val_score(
            pipeline, x, y, scoring=metric, cv=cv, n_jobs=n_jobs, error_score="raise"
        )
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"{tuning.model}-{metric}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
    )
    logger.info("Optuna: %s, %d trials, metric=%s", tuning.model, tuning.n_trials, metric)
    study.optimize(objective, n_trials=tuning.n_trials, timeout=tuning.timeout_sec)
    logger.info("Best CV %s=%.4f with %s", metric, study.best_value, study.best_params)

    pipeline = build().set_params(**study.best_params)
    pipeline.fit(x, y)
    return TuningResult(
        pipeline=pipeline,
        best_params=dict(study.best_params),
        cv_score=float(study.best_value),
        n_trials=len(study.trials),
    )
