"""
Quant-Lab Model Persistence.

Saves and loads fitted :class:`~ml.base_model.BaseModel` instances to
disk (via ``joblib``), alongside a JSON metadata sidecar describing
what was saved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib

from config.logging_config import get_logger
from config.paths import MODELS
from core.enums import ModelType
from core.exceptions import ModelError
from ml.base_model import BaseModel

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Describes a persisted model."""

    name: str
    model_type: ModelType
    feature_names: tuple[str, ...]
    trained_at: datetime

    def to_dict(self) -> dict[str, object]:
        """Serialize this metadata to a JSON-compatible ``dict``."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "feature_names": list(self.feature_names),
            "trained_at": self.trained_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ModelMetadata:
        """Deserialize metadata from a ``dict`` produced by :meth:`to_dict`."""
        feature_names = payload["feature_names"]
        if not isinstance(feature_names, list):
            raise ModelError(
                f"feature_names must be a list, got {type(feature_names).__name__}."
            )

        return cls(
            name=str(payload["name"]),
            model_type=ModelType(payload["model_type"]),
            feature_names=tuple(feature_names),
            trained_at=datetime.fromisoformat(str(payload["trained_at"])),
        )


def _model_path(model_name: str, version: str) -> Path:
    return MODELS / f"{model_name}_{version}.joblib"


def _metadata_path(model_name: str, version: str) -> Path:
    return MODELS / f"{model_name}_{version}.meta.json"


def save_model(
    model: BaseModel,
    version: str,
    *,
    feature_names: list[str] | None = None,
    trained_at: datetime | None = None,
) -> Path:
    """
    Persist a fitted model to disk, with a metadata sidecar.

    Parameters
    ----------
    model : ml.base_model.BaseModel
        The fitted model to save.
    version : str
        A version label (e.g. ``"2026-07-01"`` or ``"v1"``) used in
        the saved file names, so multiple versions of the same model
        can coexist.
    feature_names : list[str] | None, optional
        Feature names the model was trained on, recorded in the
        metadata sidecar. Defaults to an empty list.
    trained_at : datetime.datetime | None, optional
        When the model was trained. Defaults to now (UTC).

    Returns
    -------
    pathlib.Path
        The path the model was saved to.

    Raises
    ------
    core.exceptions.ModelError
        If ``model`` is not fitted.
    """
    if not model.is_fitted:
        raise ModelError(f"Cannot save {model.name}: model is not fitted yet.")

    MODELS.mkdir(parents=True, exist_ok=True)

    model_path = _model_path(model.name, version)
    joblib.dump(model, model_path)

    metadata = ModelMetadata(
        name=model.name,
        model_type=model.model_type,
        feature_names=tuple(feature_names or []),
        trained_at=trained_at or datetime.now(UTC),
    )
    _metadata_path(model.name, version).write_text(
        json.dumps(metadata.to_dict(), indent=2), encoding="utf-8"
    )

    logger.info("Saved %s (version=%s) to %s.", model.name, version, model_path)

    return model_path


def load_model(model_name: str, version: str) -> BaseModel:
    """
    Load a persisted model from disk.

    Parameters
    ----------
    model_name : str
        The model's :attr:`~ml.base_model.BaseModel.name`.
    version : str
        The version label passed to :func:`save_model`.

    Returns
    -------
    ml.base_model.BaseModel
        The deserialized, fitted model.

    Raises
    ------
    core.exceptions.ModelError
        If no saved model is found, or the file does not contain a
        :class:`~ml.base_model.BaseModel`.
    """
    model_path = _model_path(model_name, version)

    if not model_path.exists():
        raise ModelError(f"No saved model found at {model_path}.")

    model = joblib.load(model_path)

    if not isinstance(model, BaseModel):
        raise ModelError(f"File at {model_path} does not contain a BaseModel instance.")

    logger.info("Loaded %s (version=%s) from %s.", model_name, version, model_path)

    return model


def load_metadata(model_name: str, version: str) -> ModelMetadata | None:
    """
    Load the metadata sidecar for a persisted model, if it exists.

    Returns
    -------
    ModelMetadata | None
        The parsed metadata, or ``None`` if no sidecar exists.

    Raises
    ------
    core.exceptions.ModelError
        If the sidecar exists but cannot be parsed.
    """
    path = _metadata_path(model_name, version)

    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ModelMetadata.from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ModelError(f"Failed to read metadata file {path}: {exc}") from exc


def list_versions(model_name: str) -> list[str]:
    """
    List every version saved for ``model_name``, sorted.

    Returns
    -------
    list[str]
        Version labels with a saved ``.joblib`` file under
        :data:`config.paths.MODELS`.
    """
    prefix = f"{model_name}_"
    suffix = ".joblib"

    return sorted(
        path.name[len(prefix) : -len(suffix)]
        for path in MODELS.glob(f"{prefix}*{suffix}")
    )
