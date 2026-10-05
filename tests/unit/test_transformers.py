import pandas as pd
import pytest
from sklearn.base import clone

from bank_churn.exceptions import DataValidationError
from bank_churn.transformers import BalanceFlagger, FeatureSelector


def test_selector_drops_extra_columns_and_fixes_order() -> None:
    df = pd.DataFrame({"b": [1], "id": [10], "a": [2]})

    result = FeatureSelector(["a", "b"]).fit_transform(df)

    assert result.columns.tolist() == ["a", "b"]


def test_selector_reports_missing_features() -> None:
    selector = FeatureSelector(["a", "b"]).fit(pd.DataFrame({"a": [1], "b": [2]}))

    with pytest.raises(DataValidationError, match="'b'"):
        selector.transform(pd.DataFrame({"a": [1]}))


def test_balance_flag() -> None:
    df = pd.DataFrame({"Balance": [0.0, 125.5, 0.0]})

    result = BalanceFlagger().fit_transform(df)

    assert result["HasBalance"].tolist() == [0, 1, 0]
    assert "HasBalance" not in df.columns, "входной датафрейм не должен меняться"


def test_balance_flag_feature_names() -> None:
    flagger = BalanceFlagger(name="flag").fit(pd.DataFrame({"x": [1], "Balance": [0.0]}))

    assert flagger.get_feature_names_out().tolist() == ["x", "Balance", "flag"]


def test_balance_flag_requires_source_column() -> None:
    with pytest.raises(DataValidationError, match="Balance"):
        BalanceFlagger().fit(pd.DataFrame({"x": [1]}))


def test_transformers_are_cloneable() -> None:
    selector = clone(FeatureSelector(["a"]))
    flagger = clone(BalanceFlagger(source="Saldo"))

    assert selector.get_params() == {"columns": ["a"]}
    assert flagger.get_params()["source"] == "Saldo"
