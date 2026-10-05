"""Сохранение и загрузка обученной модели вместе с метаданными."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


@dataclass
class ModelBundle:
    """Всё, что нужно для инференса: пайплайн, порог и паспорт модели."""

    pipeline: Pipeline
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


def metadata_path(model_path: str | Path) -> Path:
    """Путь к JSON-паспорту модели рядом с бинарником: ``model.joblib`` → ``model.meta.json``."""
    return Path(model_path).with_suffix(".meta.json")


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Сохраняет модель (joblib) и её метаданные (человекочитаемый JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    metadata_path(path).write_text(
        json.dumps(bundle.metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    """Загружает модель. Загружайте только артефакты из доверенного источника (pickle)."""
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle (got {type(bundle).__name__})")
    return bundle
