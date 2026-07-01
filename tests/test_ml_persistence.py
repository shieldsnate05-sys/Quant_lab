"""Tests for ml.persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

import ml.persistence as persistence
from core.exceptions import ModelError
from ml.config import RandomForestConfig
from ml.dataset import ChronologicalSplit
from ml.persistence import list_versions, load_metadata, load_model, save_model
from ml.random_forest import RandomForestModel


@pytest.fixture(autouse=True)
def isolated_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ml.persistence's MODELS constant to a throwaway directory."""
    monkeypatch.setattr(persistence, "MODELS", tmp_path)
    return tmp_path


def test_save_model_rejects_unfitted_model() -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10))
    with pytest.raises(ModelError, match="not fitted"):
        save_model(model, "v1")


def test_save_and_load_round_trip(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10)).fit(
        ml_split.train.X, ml_split.train.y
    )

    path = save_model(model, "v1", feature_names=ml_split.train.feature_names)
    assert path.exists()

    loaded = load_model("RandomForest", "v1")
    assert loaded.is_fitted
    assert list(loaded.predict(ml_split.test.X)) == list(model.predict(ml_split.test.X))


def test_load_model_raises_when_missing() -> None:
    with pytest.raises(ModelError, match="No saved model"):
        load_model("RandomForest", "does-not-exist")


def test_save_model_writes_metadata_sidecar(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10)).fit(
        ml_split.train.X, ml_split.train.y
    )
    save_model(model, "v1", feature_names=ml_split.train.feature_names)

    metadata = load_metadata("RandomForest", "v1")

    assert metadata is not None
    assert metadata.name == "RandomForest"
    assert metadata.feature_names == tuple(ml_split.train.feature_names)


def test_load_metadata_returns_none_when_missing() -> None:
    assert load_metadata("RandomForest", "does-not-exist") is None


def test_list_versions(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10)).fit(
        ml_split.train.X, ml_split.train.y
    )
    save_model(model, "v1")
    save_model(model, "v2")

    assert list_versions("RandomForest") == ["v1", "v2"]


def test_list_versions_empty_when_none_saved() -> None:
    assert list_versions("RandomForest") == []
