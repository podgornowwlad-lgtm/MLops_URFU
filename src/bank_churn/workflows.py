"""Сценарии верхнего уровня: сравнение моделей, обучение, скоринг.

Каждый сценарий читает данные, вызывает доменные функции и пишет артефакты.
CLI и CI вызывают только эти функции.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bank_churn import __version__
from bank_churn.artifacts import ModelBundle, save_bundle
from bank_churn.config import ExperimentConfig, Settings
from bank_churn.data import load_dataset, split_dataset
from bank_churn.evaluation import compute_metrics, feature_importances, text_report
from bank_churn.inference import ChurnPredictor
from bank_churn.training import compare_candidates, tune_final_model

logger = logging.getLogger(__name__)

COMPARISON_FILE = "model_comparison.csv"
METRICS_FILE = "metrics.json"
REPORT_FILE = "classification_report.txt"
IMPORTANCES_FILE = "feature_importances.csv"


def run_comparison(settings: Settings, cfg: ExperimentConfig) -> pd.DataFrame:
    """Сравнивает модели-кандидаты на train и сохраняет таблицу метрик."""
    df = load_dataset(settings.data_path, cfg.columns)
    split = split_dataset(df, cfg.columns, cfg.validation)
    table = compare_candidates(split.x_train, split.y_train, cfg, n_jobs=settings.n_jobs)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.reports_dir / COMPARISON_FILE
    table.to_csv(path, index=False)
    logger.info("Comparison saved to %s\n%s", path, table.drop(columns="best_params"))
    return table


def run_training(settings: Settings, cfg: ExperimentConfig) -> dict[str, Any]:
    """Обучает финальную модель, оценивает её на отложенной выборке и сохраняет артефакты."""
    df = load_dataset(settings.data_path, cfg.columns)
    split = split_dataset(df, cfg.columns, cfg.validation)
    result = tune_final_model(split.x_train, split.y_train, cfg, n_jobs=settings.n_jobs)

    threshold = cfg.tuning.threshold
    proba = result.pipeline.predict_proba(split.x_test)[:, 1]
    metadata: dict[str, Any] = {
        "model": cfg.tuning.model,
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_source": settings.data_path,
        "n_train": len(split.x_train),
        "n_test": len(split.x_test),
        "features": [c for c in cfg.columns.features if c not in cfg.tuning.excluded_features],
        "excluded_features": cfg.tuning.excluded_features,
        "metric": cfg.validation.metric,
        "threshold": threshold,
        "best_params": result.best_params,
        "n_trials": result.n_trials,
        "cv_score": result.cv_score,
        "test_metrics": compute_metrics(split.y_test, proba, threshold),
    }

    model_path = save_bundle(
        ModelBundle(result.pipeline, threshold=threshold, metadata=metadata), settings.model_path
    )
    _write_reports(settings.reports_dir, metadata, text_report(split.y_test, proba, threshold))
    importances = feature_importances(result.pipeline)
    if importances is not None:
        importances.to_csv(settings.reports_dir / IMPORTANCES_FILE)

    logger.info("Model saved to %s; test metrics: %s", model_path, metadata["test_metrics"])
    return metadata


def run_prediction(
    settings: Settings, input_path: str | Path, output_path: Path, threshold: float | None = None
) -> pd.DataFrame:
    """Скоринг CSV сохранённой моделью; таргет во входных данных не нужен."""
    predictor = ChurnPredictor.from_file(settings.model_path)
    predictions = predictor.predict(pd.read_csv(input_path), threshold=threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("%d predictions saved to %s", len(predictions), output_path)
    return predictions


def _write_reports(reports_dir: Path, metadata: dict[str, Any], report: str) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / METRICS_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (reports_dir / REPORT_FILE).write_text(report, encoding="utf-8")
