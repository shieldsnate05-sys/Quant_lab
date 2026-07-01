"""
Quant-Lab Model Registry.

Central registry mapping model names to :class:`~ml.base_model.BaseModel`
subclasses, so training/prediction code can look up and instantiate
models by name instead of importing every concrete class directly.
"""

from __future__ import annotations

from typing import Any, TypeVar

from config.logging_config import get_logger
from core.exceptions import ModelError
from ml.base_model import BaseModel

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=type[BaseModel])


class ModelRegistry:
    """A registry of :class:`~ml.base_model.BaseModel` classes, keyed by name."""

    def __init__(self) -> None:
        self._models: dict[str, type[BaseModel]] = {}

    def register(self, model_cls: ModelT) -> ModelT:
        """
        Register ``model_cls`` under its :attr:`~ml.base_model.BaseModel.name`.

        Intended for use as a class decorator.

        Raises
        ------
        core.exceptions.ModelError
            If another model is already registered under the same name.
        """
        name = model_cls.name

        if name in self._models:
            raise ModelError(f"A model named '{name}' is already registered.")

        self._models[name] = model_cls
        logger.debug("Registered model '%s'.", name)

        return model_cls

    def get(self, name: str) -> type[BaseModel]:
        """
        Look up a registered model class by name.

        Raises
        ------
        core.exceptions.ModelError
            If no model is registered under ``name``.
        """
        try:
            return self._models[name]
        except KeyError:
            raise ModelError(
                f"No model named '{name}' is registered. Available: {self.names()}"
            ) from None

    def create(self, name: str, **kwargs: Any) -> BaseModel:
        """
        Instantiate a registered model by name.

        Parameters
        ----------
        name : str
            The registered model name.
        **kwargs
            Keyword arguments forwarded to the model's constructor.
        """
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        """Return the names of every registered model, sorted."""
        return sorted(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)


#: The global model registry. Every concrete model registers here.
MODEL_REGISTRY = ModelRegistry()
