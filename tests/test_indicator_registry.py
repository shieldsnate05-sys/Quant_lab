"""Tests for indicators.registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import pytest

from core.enums import IndicatorGroup
from core.exceptions import IndicatorError
from core.types import OHLCVFrame, PriceSeries
from indicators.base import Indicator
from indicators.registry import REGISTRY, IndicatorRegistry
from indicators.trend import EMA


def test_all_eleven_indicators_are_registered() -> None:
    expected = {
        "SMA",
        "EMA",
        "MACD",
        "PSAR",
        "RSI",
        "STOCH",
        "ATR",
        "ADX",
        "BBANDS",
        "VWAP",
        "ICHIMOKU",
    }
    assert expected <= set(REGISTRY.names())


def test_get_returns_the_registered_class() -> None:
    assert REGISTRY.get("EMA") is EMA


def test_get_raises_on_unknown_name() -> None:
    with pytest.raises(IndicatorError, match="No indicator named 'NOPE'"):
        REGISTRY.get("NOPE")


def test_create_instantiates_with_kwargs() -> None:
    indicator = REGISTRY.create("EMA", period=5)
    assert isinstance(indicator, EMA)
    assert indicator.period == 5


def test_contains_and_len() -> None:
    assert "RSI" in REGISTRY
    assert "NOPE" not in REGISTRY
    assert len(REGISTRY) >= 11


def test_register_raises_on_duplicate_name() -> None:
    registry = IndicatorRegistry()

    @dataclass(slots=True)
    class _Dummy(Indicator):
        name: ClassVar[str] = "DUMMY"
        group: ClassVar[IndicatorGroup] = IndicatorGroup.TREND

        def compute(self, frame: OHLCVFrame) -> PriceSeries:
            return frame.iloc[:, 0]

    registry.register(_Dummy)

    with pytest.raises(IndicatorError, match="already registered"):
        registry.register(_Dummy)


def test_registry_create_end_to_end(ohlcv_frame: pd.DataFrame) -> None:
    result = REGISTRY.create("SMA", period=10).compute(ohlcv_frame)
    assert isinstance(result, pd.Series)
    assert result.name == "SMA"
