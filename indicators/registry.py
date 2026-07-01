"""
Quant-Lab Indicator Registry.

Central registry mapping indicator names to :class:`~indicators.base.Indicator`
subclasses, so callers (features, strategies, reports) can look up and
instantiate indicators by name instead of importing every concrete
class directly.
"""

from __future__ import annotations

from typing import Any, TypeVar

from config.logging_config import get_logger
from core.exceptions import IndicatorError
from indicators.base import Indicator

logger = get_logger(__name__)

IndicatorT = TypeVar("IndicatorT", bound=type[Indicator])


class IndicatorRegistry:
    """A registry of :class:`~indicators.base.Indicator` classes, keyed by name."""

    def __init__(self) -> None:
        self._indicators: dict[str, type[Indicator]] = {}

    def register(self, indicator_cls: IndicatorT) -> IndicatorT:
        """
        Register ``indicator_cls`` under its :attr:`~indicators.base.Indicator.name`.

        Intended for use as a class decorator.

        Parameters
        ----------
        indicator_cls : type[indicators.base.Indicator]
            The indicator class to register.

        Returns
        -------
        type[indicators.base.Indicator]
            ``indicator_cls``, unchanged, so this can be used as a decorator.

        Raises
        ------
        core.exceptions.IndicatorError
            If another indicator is already registered under the same name.
        """
        name = indicator_cls.name

        if name in self._indicators:
            raise IndicatorError(f"An indicator named '{name}' is already registered.")

        self._indicators[name] = indicator_cls
        logger.debug("Registered indicator '%s'.", name)

        return indicator_cls

    def get(self, name: str) -> type[Indicator]:
        """
        Look up a registered indicator class by name.

        Raises
        ------
        core.exceptions.IndicatorError
            If no indicator is registered under ``name``.
        """
        try:
            return self._indicators[name]
        except KeyError:
            raise IndicatorError(
                f"No indicator named '{name}' is registered. "
                f"Available: {self.names()}"
            ) from None

    def create(self, name: str, **kwargs: Any) -> Indicator:
        """
        Instantiate a registered indicator by name.

        Parameters
        ----------
        name : str
            The registered indicator name.
        **kwargs
            Keyword arguments forwarded to the indicator's constructor.

        Returns
        -------
        indicators.base.Indicator
            A new instance of the registered indicator.

        Raises
        ------
        core.exceptions.IndicatorError
            If no indicator is registered under ``name``.
        """
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        """Return the names of every registered indicator, sorted."""
        return sorted(self._indicators)

    def __contains__(self, name: str) -> bool:
        return name in self._indicators

    def __len__(self) -> int:
        return len(self._indicators)


#: The global indicator registry. Every concrete indicator registers here.
REGISTRY = IndicatorRegistry()
