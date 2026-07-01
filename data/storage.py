"""
Quant-Lab Parquet Storage.

Low-level, path-based Parquet read/write primitives. This is the only
module in the platform that touches ``pandas.read_parquet`` /
``DataFrame.to_parquet`` directly - higher-level modules
(:mod:`data.cache`) build key-based caching semantics on top of it
instead of duplicating I/O and error-handling logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.logging_config import get_logger
from core.exceptions import DataError
from core.types import OHLCVFrame

logger = get_logger(__name__)


class ParquetStorage:
    """
    Reads and writes DataFrames to Parquet files under a base directory.

    Parameters
    ----------
    base_dir : pathlib.Path
        Root directory all resolved paths are relative to. Created if
        it does not already exist.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        """The root directory this storage instance reads from and writes to."""
        return self._base_dir

    def resolve_path(self, filename: str) -> Path:
        """
        Resolve ``filename`` to an absolute path under :attr:`base_dir`.

        Parameters
        ----------
        filename : str
            File name (or relative path) to resolve.

        Returns
        -------
        pathlib.Path
            The resolved absolute path.
        """
        return self._base_dir / filename

    def exists(self, path: Path) -> bool:
        """Return ``True`` if a Parquet file exists at ``path``."""
        return path.exists()

    def read(self, path: Path) -> OHLCVFrame:
        """
        Read a DataFrame from the Parquet file at ``path``.

        Parameters
        ----------
        path : pathlib.Path
            Path to the Parquet file.

        Returns
        -------
        core.types.OHLCVFrame
            The deserialized DataFrame.

        Raises
        ------
        core.exceptions.DataError
            If ``path`` does not exist or cannot be read.
        """
        if not path.exists():
            raise DataError(f"Parquet file does not exist: {path}")

        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataError(f"Failed to read Parquet file {path}: {exc}") from exc

        logger.debug("Read %d rows from %s.", len(frame), path)

        return frame

    def write(self, path: Path, frame: pd.DataFrame) -> None:
        """
        Write ``frame`` to a Parquet file at ``path``.

        Parameters
        ----------
        path : pathlib.Path
            Destination path. Parent directories are created if needed.
        frame : pandas.DataFrame
            The DataFrame to serialize.

        Raises
        ------
        core.exceptions.DataError
            If ``frame`` cannot be written to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            frame.to_parquet(path)
        except Exception as exc:
            raise DataError(f"Failed to write Parquet file {path}: {exc}") from exc

        logger.debug("Wrote %d rows to %s.", len(frame), path)

    def delete(self, path: Path) -> None:
        """Delete the Parquet file at ``path``, if it exists."""
        if path.exists():
            path.unlink()
            logger.debug("Deleted %s.", path)
