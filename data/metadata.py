"""
Quant-Lab Dataset Metadata.

Describes a cached OHLCV dataset (symbol, timeframe, coverage, source,
and fetch time) and persists that description as a JSON sidecar file
next to the Parquet data it describes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config.logging_config import get_logger
from core.enums import TimeFrame
from core.exceptions import DataError
from core.types import OHLCVFrame, Symbol

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Describes a single cached OHLCV dataset."""

    symbol: Symbol
    timeframe: TimeFrame
    start: datetime
    end: datetime
    row_count: int
    source: str
    fetched_at: datetime

    def to_dict(self) -> dict[str, str | int]:
        """Serialize this metadata to a JSON-compatible ``dict``."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "row_count": self.row_count,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str | int]) -> DatasetMetadata:
        """Deserialize metadata from a ``dict`` produced by :meth:`to_dict`."""
        return cls(
            symbol=str(payload["symbol"]),
            timeframe=TimeFrame(payload["timeframe"]),
            start=datetime.fromisoformat(str(payload["start"])),
            end=datetime.fromisoformat(str(payload["end"])),
            row_count=int(payload["row_count"]),
            source=str(payload["source"]),
            fetched_at=datetime.fromisoformat(str(payload["fetched_at"])),
        )


def build_metadata(
    symbol: Symbol,
    timeframe: TimeFrame,
    frame: OHLCVFrame,
    *,
    source: str,
    fetched_at: datetime,
) -> DatasetMetadata:
    """
    Build :class:`DatasetMetadata` describing ``frame``.

    Parameters
    ----------
    symbol : core.types.Symbol
        Ticker symbol the frame covers.
    timeframe : core.enums.TimeFrame
        Bar timeframe of the frame.
    frame : core.types.OHLCVFrame
        The OHLCV frame to describe. Must not be empty.
    source : str
        Identifier for the data source (e.g. ``"alpaca"``).
    fetched_at : datetime.datetime
        The time the data was fetched.

    Returns
    -------
    DatasetMetadata
        Metadata describing ``frame``.

    Raises
    ------
    core.exceptions.DataError
        If ``frame`` is empty.
    """
    if frame.empty:
        raise DataError("Cannot build metadata for an empty OHLCV frame.")

    return DatasetMetadata(
        symbol=symbol,
        timeframe=timeframe,
        start=frame.index[0].to_pydatetime(),
        end=frame.index[-1].to_pydatetime(),
        row_count=len(frame),
        source=source,
        fetched_at=fetched_at,
    )


def write_metadata(path: Path, metadata: DatasetMetadata) -> None:
    """
    Write ``metadata`` to ``path`` as JSON.

    Raises
    ------
    core.exceptions.DataError
        If ``metadata`` cannot be written to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    except OSError as exc:
        raise DataError(f"Failed to write metadata file {path}: {exc}") from exc

    logger.debug(
        "Wrote metadata for %s (%s) to %s.",
        metadata.symbol,
        metadata.timeframe.value,
        path,
    )


def read_metadata(path: Path) -> DatasetMetadata | None:
    """
    Read :class:`DatasetMetadata` from ``path``, if it exists.

    Returns
    -------
    DatasetMetadata | None
        The parsed metadata, or ``None`` if ``path`` does not exist.

    Raises
    ------
    core.exceptions.DataError
        If ``path`` exists but cannot be parsed.
    """
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DatasetMetadata.from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise DataError(f"Failed to read metadata file {path}: {exc}") from exc


__all__ = [
    "DatasetMetadata",
    "build_metadata",
    "read_metadata",
    "write_metadata",
]
