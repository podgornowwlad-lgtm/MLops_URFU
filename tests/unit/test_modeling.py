import pandas as pd
import pytest

from bank_churn.config import ExperimentConfig
from bank_churn.modeling import MODEL_REGISTRY, build_pipeline


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_every_model_fits_raw_data(
    model_name: str, churn_df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    pipeline = build_pipeline(model_name, experiment.columns, seed=0)
    if model_name == "catboost":
        pipeline.set_params(model__iterations=10)
    x, y = churn_df.drop(columns="Exited"), churn_df["Exited"]

    proba = pipeline.fit(x, y).predict_proba(x)

    assert proba.shape == (len(x), 2)


def test_linear_pipeline_has_scaling_steps(experiment: ExperimentConfig) -> None:
    linear = build_pipeline("logreg", experiment.columns)
    tree = build_pipeline("decision_tree", experiment.columns)

    assert {"poly", "scale"} <= set(linear.named_steps)
    assert not {"poly", "scale"} & set(tree.named_steps)


def test_excluded_features_do_not_reach_model(
    churn_df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    excluded = ["HasCrCard", "EstimatedSalary", "Tenure"]
    pipeline = build_pipeline("decision_tree", experiment.columns, exclude=excluded)

    pipeline.fit(churn_df.drop(columns="Exited"), churn_df["Exited"])
    names = set(pipeline[:-1].get_feature_names_out())

    assert not names & set(excluded)
    assert {"HasBalance", "Geography_Germany", "Gender_Male"} <= names


def test_balance_flag_is_skipped_without_balance(experiment: ExperimentConfig) -> None:
    pipeline = build_pipeline("decision_tree", experiment.columns, exclude=["Balance"])

    assert "balance_flag" not in pipeline.named_steps


def test_unknown_model_is_rejected(experiment: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="unknown model 'svm'"):
        build_pipeline("svm", experiment.columns)
