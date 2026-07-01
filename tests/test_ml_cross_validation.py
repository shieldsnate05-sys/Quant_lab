"""Tests for ml.cross_validation."""

from __future__ import annotations

import pytest

from core.exceptions import ModelError
from ml.config import RandomForestConfig
from ml.cross_validation import cross_validate_model, make_time_series_split
from ml.dataset import Dataset
from ml.random_forest import RandomForestModel


def test_make_time_series_split_rejects_too_few_splits() -> None:
    with pytest.raises(ModelError, match="n_splits"):
        make_time_series_split(1)


def test_make_time_series_split_produces_chronological_folds(
    ml_dataset: Dataset,
) -> None:
    splitter = make_time_series_split(3)
    folds = list(splitter.split(ml_dataset.X))

    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert max(train_idx) < min(test_idx)


def test_cross_validate_model_produces_one_metrics_entry_per_fold(
    ml_dataset: Dataset,
) -> None:
    result = cross_validate_model(
        lambda: RandomForestModel(RandomForestConfig(n_estimators=20)),
        ml_dataset,
        n_splits=3,
    )

    assert len(result.fold_metrics) == 3
    assert 0 <= result.mean_accuracy <= 1
    assert 0 <= result.mean_f1 <= 1


def test_cross_validate_model_uses_fresh_model_per_fold(ml_dataset: Dataset) -> None:
    built_models = []

    def factory() -> RandomForestModel:
        model = RandomForestModel(RandomForestConfig(n_estimators=10))
        built_models.append(model)
        return model

    cross_validate_model(factory, ml_dataset, n_splits=3)

    assert len(built_models) == 3
    assert len({id(m) for m in built_models}) == 3
