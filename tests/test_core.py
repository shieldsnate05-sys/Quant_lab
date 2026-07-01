"""Tests for the core package: exceptions, enums, constants, types."""

from __future__ import annotations

import pytest

from core.constants import EMA_FAST, EMA_SLOW, RSI_PERIOD
from core.enums import PositionSide, TimeFrame
from core.exceptions import (
    BacktestError,
    DataError,
    IndicatorError,
    QuantLabError,
    StrategyError,
)


@pytest.mark.parametrize(
    "exception_cls",
    [DataError, IndicatorError, StrategyError, BacktestError],
)
def test_exceptions_derive_from_quant_lab_error(
    exception_cls: type[QuantLabError],
) -> None:
    assert issubclass(exception_cls, QuantLabError)
    assert issubclass(exception_cls, Exception)


def test_quant_lab_error_is_catchable_as_exception() -> None:
    with pytest.raises(QuantLabError):
        raise DataError("boom")


def test_timeframe_values() -> None:
    assert TimeFrame.DAILY.value == "1Day"
    assert TimeFrame.ONE_MINUTE.value == "1Min"


def test_position_side_values() -> None:
    assert {side.value for side in PositionSide} == {"LONG", "SHORT", "FLAT"}


def test_ema_constants_are_ordered() -> None:
    assert EMA_FAST < EMA_SLOW
    assert RSI_PERIOD > 0
