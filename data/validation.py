"""
Quant-Lab OHLCV Validation.

Structural and sanity validation for OHLCV market data frames. Shared
by every package that consumes market data (``indicators``,
``features``, ``strategies``, ``backtesting``) so validation logic is
defined exactly once.
"""

from __future__ import annotations

import pandas as pd

from core.exceptions import DataError
from core.types import OHLCVFrame
from data.schema import SCHEMA, OHLCVSchema


def validate_ohlcv_frame(
    frame: OHLCVFrame,
    *,
    schema: OHLCVSchema = SCHEMA,
    allow_empty: bool = False,
) -> None:
    """
    Validate that ``frame`` is a well-formed OHLCV market data frame.

    Parameters
    ----------
    frame : core.types.OHLCVFrame
        The market data frame to validate.
    schema : data.schema.OHLCVSchema, optional
        The column schema to validate against. Defaults to
        :data:`data.schema.SCHEMA`.
    allow_empty : bool, optional
        If ``False`` (the default), an empty frame raises
        :class:`~core.exceptions.DataError`.

    Raises
    ------
    core.exceptions.DataError
        If ``frame`` is missing required columns, has a non-datetime or
        non-monotonic index, contains duplicate timestamps, contains
        null values in required columns, or contains rows that violate
        OHLC price-ordering invariants (``low <= open, close <= high``)
        or negative volume.
    """
    if frame.empty:
        if allow_empty:
            return
        raise DataError("OHLCV frame is empty.")

    missing_columns = set(schema.columns) - set(frame.columns)
    if missing_columns:
        raise DataError(
            f"OHLCV frame is missing required columns: {sorted(missing_columns)}"
        )

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataError("OHLCV frame must be indexed by a DatetimeIndex.")

    if not frame.index.is_monotonic_increasing:
        raise DataError("OHLCV frame index must be sorted in ascending order.")

    if frame.index.has_duplicates:
        raise DataError("OHLCV frame index contains duplicate timestamps.")

    required = list(schema.columns)
    if frame[required].isnull().any().any():
        raise DataError("OHLCV frame contains null values in required columns.")

    if (frame[schema.high] < frame[schema.low]).any():
        raise DataError("OHLCV frame contains rows where high < low.")

    if (
        (frame[schema.high] < frame[schema.open]).any()
        or (frame[schema.high] < frame[schema.close]).any()
        or (frame[schema.low] > frame[schema.open]).any()
        or (frame[schema.low] > frame[schema.close]).any()
    ):
        raise DataError(
            "OHLCV frame contains rows where open/close fall outside [low, high]."
        )

    if (frame[schema.volume] < 0).any():
        raise DataError("OHLCV frame contains negative volume.")
