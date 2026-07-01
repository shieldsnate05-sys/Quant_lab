"""Tests for ml.feature_importance."""

from __future__ import annotations

import pytest

from core.exceptions import ModelError
from ml.config import RandomForestConfig
from ml.dataset import ChronologicalSplit
from ml.feature_importance import built_in_importance, permutation_feature_importance
from ml.random_forest import RandomForestModel


def test_built_in_importance_matches_model_method(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20)).fit(
        ml_split.train.X, ml_split.train.y
    )
    result = built_in_importance(model)
    assert set(result.index) == set(ml_split.train.feature_names)


def test_permutation_feature_importance_covers_every_feature(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20)).fit(
        ml_split.train.X, ml_split.train.y
    )

    result = permutation_feature_importance(model, ml_split.test, n_repeats=3)

    assert set(result.index) == set(ml_split.test.feature_names)
    assert list(result.values) == sorted(result.values, reverse=True)


def test_permutation_feature_importance_rejects_unfitted_model(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20))
    with pytest.raises(ModelError, match="not fitted"):
        permutation_feature_importance(model, ml_split.test)
