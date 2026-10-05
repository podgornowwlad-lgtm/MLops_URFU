"""Загрузка, проверка схемы и разбиение датасета."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from bank_churn.config import ColumnsConfig, ValidationConfig
from bank_churn.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class DatasetSplit(NamedTuple):
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def validate_frame(df: pd.DataFrame, columns: ColumnsConfig, *, with_target: bool) -> None:
    """Проверяет, что в датафрейме есть все признаки (и корректный бинарный таргет)."""
    if df.empty:
        raise DataValidationError("dataset is empty")

    required = [*columns.features, *([columns.target] if with_target else [])]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"dataset is missing required columns: {missing}")

    if with_target:
        target = df[columns.target]
        if target.isna().any():
            raise DataValidationError(f"target {columns.target!r} contains missing values")
        unexpected = sorted(set(target.unique()) - {0, 1})
        if unexpected:
            raise DataValidationError(
                f"target {columns.target!r} must be binary 0/1, got extra values {unexpected}"
            )


def load_dataset(
    source: str | Path, columns: ColumnsConfig, *, with_target: bool = True
) -> pd.DataFrame:
    """Читает CSV (локальный путь или URL) и проверяет схему."""
    logger.info("Reading dataset from %s", source)
    df = pd.read_csv(source)
    validate_frame(df, columns, with_target=with_target)
    logger.info("Dataset loaded: %d rows x %d columns", *df.shape)
    return df


def split_dataset(
    df: pd.DataFrame, columns: ColumnsConfig, validation: ValidationConfig
) -> DatasetSplit:
    """Стратифицированное разбиение на обучающую и отложенную выборки."""
    x = df.drop(columns=[columns.target])
    y = df[columns.target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=validation.test_size,
        stratify=y,
        random_state=validation.split_seed,
    )
    logger.info("Split: train=%d, test=%d, churn rate=%.3f", len(x_train), len(x_test), y.mean())
    return DatasetSplit(x_train, x_test, y_train, y_test)
