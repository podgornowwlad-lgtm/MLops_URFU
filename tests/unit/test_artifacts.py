import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from bank_churn.artifacts import ModelBundle, load_bundle, metadata_path, save_bundle
from bank_churn.config import ExperimentConfig
from bank_churn.inference import ChurnPredictor
from bank_churn.modeling import build_pipeline


@pytest.fixture
def bundle(churn_df: pd.DataFrame, experiment: ExperimentConfig) -> ModelBundle:
    pipeline = build_pipeline("decision_tree", experiment.columns, seed=0)
    pipeline.set_params(model__max_depth=3)
    pipeline.fit(churn_df.drop(columns="Exited"), churn_df["Exited"])
    return ModelBundle(pipeline, threshold=0.4, metadata={"model": "decision_tree"})


def test_bundle_roundtrip(tmp_path: Path, bundle: ModelBundle) -> None:
    path = save_bundle(bundle, tmp_path / "nested" / "model.joblib")

    loaded = load_bundle(path)

    assert loaded.threshold == 0.4
    meta = json.loads(metadata_path(path).read_text(encoding="utf-8"))
    assert meta == {"model": "decision_tree"}


def test_load_rejects_foreign_objects(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump({"not": "a bundle"}, path)

    with pytest.raises(TypeError, match="ModelBundle"):
        load_bundle(path)


def test_predictor_on_raw_rows(bundle: ModelBundle, churn_df: pd.DataFrame) -> None:
    raw = churn_df.drop(columns="Exited").sample(5, random_state=1)

    result = ChurnPredictor(bundle).predict(raw)

    assert result.columns.tolist() == ["id", "churn_probability", "churn_flag"]
    assert result["id"].tolist() == raw["id"].tolist()
    assert result["churn_probability"].between(0, 1).all()
    expected = (result["churn_probability"] >= 0.4).astype(int)
    assert result["churn_flag"].tolist() == expected.tolist()


def test_predictor_without_technical_columns(bundle: ModelBundle, churn_df: pd.DataFrame) -> None:
    raw = churn_df.drop(columns=["Exited", "id", "CustomerId", "Surname"]).head(3)

    result = ChurnPredictor(bundle).predict(raw, threshold=1.0)

    assert result.columns.tolist() == ["churn_probability", "churn_flag"]
    assert len(result) == 3
