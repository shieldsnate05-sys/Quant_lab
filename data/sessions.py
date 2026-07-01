"""
Quant-Lab Trading Session Utilities.

Filters OHLCV bars down to regular trading-session hours and weekdays.

These utilities only account for time-of-day and day-of-week; they do
not implement an exchange holiday calendar. Callers that need holidays
excluded must filter those separately.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import cast

import pandas as pd

from config.logging_config import get_logger
from config.settings import settings
from core.constants import MARKET_CLOSE, MARKET_OPEN
from core.exceptions import DataError
from core.types import OHLCVFrame
from data.validator import validate_ohlcv_frame

logger = get_logger(__name__)


def regular_session_mask(
    index: pd.DatetimeIndex,
    *,
    timezone: str = settings.data.timezone,
    market_open: time = MARKET_OPEN,
    market_close: time = MARKET_CLOSE,
) -> pd.Series:
    """
    Build a boolean mask of ``index`` entries within regular trading hours.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timezone-aware timestamps to evaluate.
    timezone : str, optional
        IANA timezone the session boundaries are defined in. Defaults
        to :data:`config.settings.settings.data.timezone`.
    market_open, market_close : datetime.time, optional
        Regular session boundaries, inclusive. Default to
        :data:`core.constants.MARKET_OPEN` / :data:`core.constants.MARKET_CLOSE`.

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``index``.

    Raises
    ------
    core.exceptions.DataError
        If ``index`` is timezone-naive.
    """
    if index.tz is None:
        raise DataError("Cannot compute a session mask on a timezone-naive index.")

    local_times = index.tz_convert(timezone).time

    return pd.Series(
        [market_open <= t <= market_close for t in local_times],
        index=index,
        name="regular_session",
    )


def is_trading_day(
    timestamp: datetime, *, timezone: str = settings.data.timezone
) -> bool:
    """
    Return ``True`` if ``timestamp`` falls on a weekday (Monday-Friday).

    Parameters
    ----------
    timestamp : datetime.datetime
        A timezone-aware timestamp.
    timezone : str, optional
        IANA timezone to evaluate the day-of-week in. Defaults to
        :data:`config.settings.settings.data.timezone`.

    Returns
    -------
    bool
        ``True`` for Monday through Friday, ``False`` for weekends.
        Does not account for exchange holidays.

    Raises
    ------
    core.exceptions.DataError
        If ``timestamp`` is timezone-naive.
    """
    if timestamp.tzinfo is None:
        raise DataError("Cannot evaluate trading day for a timezone-naive timestamp.")

    return pd.Timestamp(timestamp).tz_convert(timezone).weekday() < 5


def filter_regular_session(
    frame: OHLCVFrame,
    *,
    timezone: str = settings.data.timezone,
    market_open: time = MARKET_OPEN,
    market_close: time = MARKET_CLOSE,
) -> OHLCVFrame:
    """
    Filter ``frame`` down to bars within regular trading hours on weekdays.

    Parameters
    ----------
    frame : core.types.OHLCVFrame
        OHLCV bars conforming to :data:`data.schema.SCHEMA`.
    timezone : str, optional
        IANA timezone the session boundaries are defined in. Defaults
        to :data:`config.settings.settings.data.timezone`.
    market_open, market_close : datetime.time, optional
        Regular session boundaries, inclusive. Default to
        :data:`core.constants.MARKET_OPEN` / :data:`core.constants.MARKET_CLOSE`.

    Returns
    -------
    core.types.OHLCVFrame
        The subset of ``frame`` falling within regular trading hours on
        weekdays. Does not exclude exchange holidays.
    """
    validate_ohlcv_frame(frame)

    # validate_ohlcv_frame already confirmed this is a tz-aware DatetimeIndex.
    index = cast(pd.DatetimeIndex, frame.index)

    time_mask = regular_session_mask(
        index, timezone=timezone, market_open=market_open, market_close=market_close
    )
    weekday_mask = pd.Series(index.tz_convert(timezone).weekday < 5, index=index)
    filtered = frame.loc[time_mask & weekday_mask]

    logger.info(
        "Filtered %d bars down to %d regular-session bars.", len(frame), len(filtered)
    )

    return filtered
