"""
Quant-Lab OHLCV Resampler.

Aggregates OHLCV bars from a finer timeframe to a coarser one (e.g.
1-minute bars to 5-minute or daily bars), using the standard OHLC
aggregation rule: first open, max high, min low, last close, summed
volume.
"""

from __future__ import annotations

from config.logging_config import get_logger
from core.enums import TimeFrame
from core.exceptions import DataError
from core.types import OHLCVFrame
from data.schema import SCHEMA, OHLCVSchema
from data.validator import validate_ohlcv_frame

logger = get_logger(__name__)

#: Approximate bar duration in seconds, used to determine whether a
#: resample target is coarser (downsample) or finer (upsample) than the source.
_APPROX_SECONDS: dict[TimeFrame, int] = {
    TimeFrame.ONE_MINUTE: 60,
    TimeFrame.FIVE_MINUTE: 5 * 60,
    TimeFrame.FIFTEEN_MINUTE: 15 * 60,
    TimeFrame.THIRTY_MINUTE: 30 * 60,
    TimeFrame.ONE_HOUR: 60 * 60,
    TimeFrame.DAILY: 24 * 60 * 60,
}

#: Pandas resample offset alias for each timeframe.
_PANDAS_FREQ: dict[TimeFrame, str] = {
    TimeFrame.ONE_MINUTE: "1min",
    TimeFrame.FIVE_MINUTE: "5min",
    TimeFrame.FIFTEEN_MINUTE: "15min",
    TimeFrame.THIRTY_MINUTE: "30min",
    TimeFrame.ONE_HOUR: "1h",
    TimeFrame.DAILY: "1D",
}


def resample_ohlcv(
    frame: OHLCVFrame,
    source_timeframe: TimeFrame,
    target_timeframe: TimeFrame,
    *,
    schema: OHLCVSchema = SCHEMA,
) -> OHLCVFrame:
    """
    Resample OHLCV ``frame`` from ``source_timeframe`` to ``target_timeframe``.

    Parameters
    ----------
    frame : core.types.OHLCVFrame
        OHLCV bars at ``source_timeframe`` resolution, conforming to
        ``schema``.
    source_timeframe : core.enums.TimeFrame
        The timeframe ``frame`` is currently sampled at.
    target_timeframe : core.enums.TimeFrame
        The timeframe to resample to. Must be coarser than
        ``source_timeframe``.
    schema : data.schema.OHLCVSchema, optional
        The column schema to aggregate. Defaults to
        :data:`data.schema.SCHEMA`.

    Returns
    -------
    core.types.OHLCVFrame
        OHLCV bars resampled to ``target_timeframe``. Bars with no
        source data (e.g. non-trading periods) are dropped.

    Raises
    ------
    core.exceptions.DataError
        If ``target_timeframe`` is not coarser than ``source_timeframe``,
        or ``frame`` fails validation.
    """
    validate_ohlcv_frame(frame, schema=schema)

    if _APPROX_SECONDS[target_timeframe] <= _APPROX_SECONDS[source_timeframe]:
        raise DataError(
            f"Cannot resample from {source_timeframe.value} to "
            f"{target_timeframe.value}: target must be coarser than source."
        )

    aggregation = {
        schema.open: "first",
        schema.high: "max",
        schema.low: "min",
        schema.close: "last",
        schema.volume: "sum",
    }

    resampled = (
        frame.resample(_PANDAS_FREQ[target_timeframe])
        .agg(aggregation)  # type: ignore[arg-type]  # pandas-stubs rejects this valid dict form
        .dropna(how="any")
    )
    resampled.index.name = schema.timestamp

    validate_ohlcv_frame(resampled, schema=schema, allow_empty=True)

    logger.info(
        "Resampled %d bars (%s) into %d bars (%s).",
        len(frame),
        source_timeframe.value,
        len(resampled),
        target_timeframe.value,
    )

    return resampled
