"""Tests for indicators.volatility: ATR, ADX, BollingerBands."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from core.exceptions import IndicatorError
from indicators.base import Indicator
from indicators.volatility import ADX, ATR, BollingerBands

IndicatorFactory = Callable[..., Indicator]


def test_atr_is_non_negative(ohlcv_frame: pd.DataFrame) -> None:
    result = ATR().compute(ohlcv_frame)
    valid = result.dropna()
    assert not valid.empty
    assert (valid >= 0).all()
    assert result.name == "ATR"


def test_adx_bounds_and_columns(ohlcv_frame: pd.DataFrame) -> None:
    result = ADX().compute(ohlcv_frame)

    assert list(result.columns) == ["adx", "plus_di", "minus_di"]
    valid = result.dropna()
    assert not valid.empty
    assert (valid["adx"] >= 0).all()
    assert (valid["adx"] <= 100).all()


def test_bollinger_bands_ordering(ohlcv_frame: pd.DataFrame) -> None:
    result = BollingerBands().compute(ohlcv_frame)

    assert list(result.columns) == ["upper", "middle", "lower"]
    valid = result.dropna()
    assert not valid.empty
    assert (valid["upper"] >= valid["middle"]).all()
    assert (valid["middle"] >= valid["lower"]).all()


def test_bollinger_bands_widen_with_larger_num_std(ohlcv_frame: pd.DataFrame) -> None:
    narrow = BollingerBands(num_std=1).compute(ohlcv_frame).dropna()
    wide = BollingerBands(num_std=3).compute(ohlcv_frame).dropna()

    assert (
        (wide["upper"] - wide["lower"]) >= (narrow["upper"] - narrow["lower"])
    ).all()


@pytest.mark.parametrize("indicator_cls", [ATR, ADX, BollingerBands])
def test_indicator_rejects_insufficient_rows(
    indicator_cls: IndicatorFactory, ohlcv_frame: pd.DataFrame
) -> None:
    with pytest.raises(IndicatorError):
        indicator_cls(period=1000).compute(ohlcv_frame)
