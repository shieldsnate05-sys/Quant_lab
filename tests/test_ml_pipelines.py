"""Tests for ml.pipelines."""

from __future__ import annotations

from pathlib import Path

import pytest

import ml.persistence as persistence
from ml.config import CalibrationConfig, RandomForestConfig
from ml.dataset import ChronologicalSplit
from ml.pipelines import TrainingPipeline
from ml.random_forest import RandomForestModel


@pytest.fixture(autouse=True)
def isolated_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ml.persistence's MODELS constant to a throwaway directory."""
    monkeypatch.setattr(persistence, "MODELS", tmp_path)
    return tmp_path


def test_pipeline_without_calibration_or_persistence(
    ml_split: ChronologicalSplit,
) -> None:
    pipeline = TrainingPipeline(
        model=RandomForestModel(RandomForestConfig(n_estimators=20))
    )

    result = pipeline.run(ml_split)

    assert result.model.is_fitted
    assert result.saved_path is None
    assert 0 <= result.evaluation.metrics.accuracy <= 1
    assert not result.evaluation.feature_importances.empty


def test_pipeline_with_calibration_has_empty_importances(
    ml_split: ChronologicalSplit,
) -> None:
    pipeline = TrainingPipeline(
        model=RandomForestModel(RandomForestConfig(n_estimators=20)),
        calibrate=True,
        calibration_config=CalibrationConfig(cv=3),
    )

    result = pipeline.run(ml_split)

    assert result.evaluation.feature_importances.empty


def test_pipeline_persists_model_when_version_given(
    ml_split: ChronologicalSplit,
) -> None:
    pipeline = TrainingPipeline(
        model=RandomForestModel(RandomForestConfig(n_estimators=20))
    )

    result = pipeline.run(ml_split, model_version="v1")

    assert result.saved_path is not None
    assert result.saved_path.exists()


def test_pipeline_preprocessor_is_fitted_only_on_training_split(
    ml_split: ChronologicalSplit,
) -> None:
    pipeline = TrainingPipeline(
        model=RandomForestModel(RandomForestConfig(n_estimators=20))
    )

    result = pipeline.run(ml_split)

    assert result.preprocessor.is_fitted
