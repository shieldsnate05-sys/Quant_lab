"""Tests for ml.walk_forward."""

from __future__ import annotations

import pytest

from core.exceptions import ModelError
from ml.config import RandomForestConfig, WalkForwardConfig
from ml.dataset import Dataset
from ml.random_forest import RandomForestModel
from ml.walk_forward import run_walk_forward


def test_run_walk_forward_produces_expected_number_of_windows(
    ml_dataset: Dataset,
) -> None:
    config = WalkForwardConfig(train_window=150, test_window=50, step=50)

    result = run_walk_forward(
        lambda: RandomForestModel(RandomForestConfig(n_estimators=15)),
        ml_dataset,
        config,
    )

    n = len(ml_dataset)
    expected_windows = (n - config.train_window) // config.test_window
    assert len(result.windows) == expected_windows


def test_run_walk_forward_windows_are_chronological(ml_dataset: Dataset) -> None:
    config = WalkForwardConfig(train_window=150, test_window=50, step=50)
    result = run_walk_forward(
        lambda: RandomForestModel(RandomForestConfig(n_estimators=15)),
        ml_dataset,
        config,
    )

    for window in result.windows:
        assert window.train_end < window.test_start
        assert window.train_start <= window.train_end
        assert window.test_start <= window.test_end


def test_run_walk_forward_expanding_grows_training_window(ml_dataset: Dataset) -> None:
    config = WalkForwardConfig(
        train_window=100, test_window=50, step=50, expanding=True
    )
    result = run_walk_forward(
        lambda: RandomForestModel(RandomForestConfig(n_estimators=15)),
        ml_dataset,
        config,
    )

    assert result.windows[0].train_start == result.windows[-1].train_start
    assert result.windows[-1].train_end > result.windows[0].train_end


def test_out_of_sample_predictions_cover_every_test_bar(ml_dataset: Dataset) -> None:
    config = WalkForwardConfig(train_window=150, test_window=50, step=50)
    result = run_walk_forward(
        lambda: RandomForestModel(RandomForestConfig(n_estimators=15)),
        ml_dataset,
        config,
    )

    total_test_rows = sum(len(w.predictions) for w in result.windows)
    assert len(result.out_of_sample_predictions) == total_test_rows


def test_run_walk_forward_rejects_dataset_smaller_than_one_window(
    ml_dataset: Dataset,
) -> None:
    config = WalkForwardConfig(train_window=10_000, test_window=1)
    with pytest.raises(ModelError, match="fewer than"):
        run_walk_forward(
            lambda: RandomForestModel(RandomForestConfig()), ml_dataset, config
        )


def test_mean_metrics_are_bounded(ml_dataset: Dataset) -> None:
    config = WalkForwardConfig(train_window=150, test_window=50, step=50)
    result = run_walk_forward(
        lambda: RandomForestModel(RandomForestConfig(n_estimators=15)),
        ml_dataset,
        config,
    )

    assert 0 <= result.mean_accuracy <= 1
    assert 0 <= result.mean_f1 <= 1
