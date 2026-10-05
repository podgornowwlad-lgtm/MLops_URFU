from pathlib import Path

import pandas as pd
import pytest

from bank_churn.config import ExperimentConfig
from bank_churn.data import load_dataset, split_dataset, validate_frame
from bank_churn.exceptions import DataValidationError


def test_load_dataset_roundtrip(
    tmp_path: Path, churn_df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    path = tmp_path / "train.csv"
    churn_df.to_csv(path, index=False)

    df = load_dataset(path, experiment.columns)

    assert df.shape == churn_df.shape


def test_missing_feature_is_reported(churn_df: pd.DataFrame, experiment: ExperimentConfig) -> None:
    with pytest.raises(DataValidationError, match="Balance"):
        validate_frame(churn_df.drop(columns="Balance"), experiment.columns, with_target=True)


def test_target_is_optional_for_inference(
    churn_df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    validate_frame(churn_df.drop(columns="Exited"), experiment.columns, with_target=False)


@pytest.mark.parametrize("bad_value", [2, None])
def test_target_must_be_binary(
    churn_df: pd.DataFrame, experiment: ExperimentConfig, bad_value: int | None
) -> None:
    churn_df["Exited"] = churn_df["Exited"].astype(object)
    churn_df.loc[0, "Exited"] = bad_value

    with pytest.raises(DataValidationError, match="Exited"):
        validate_frame(churn_df, experiment.columns, with_target=True)


def test_empty_frame_is_rejected(churn_df: pd.DataFrame, experiment: ExperimentConfig) -> None:
    with pytest.raises(DataValidationError, match="empty"):
        validate_frame(churn_df.iloc[:0], experiment.columns, with_target=True)


def test_split_is_stratified(churn_df: pd.DataFrame, experiment: ExperimentConfig) -> None:
    split = split_dataset(churn_df, experiment.columns, experiment.validation)

    assert "Exited" not in split.x_train.columns
    assert len(split.x_test) == pytest.approx(0.2 * len(churn_df), abs=1)
    assert split.y_train.mean() == pytest.approx(split.y_test.mean(), abs=0.03)
