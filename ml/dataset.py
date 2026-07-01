"""
Quant-Lab ML Dataset Assembly.

Builds labeled datasets for classification from OHLCV bars and
engineered features, and splits them chronologically. Splits are
always positional (by row order), never random or shuffled - a random
split would leak future information into the training set for time
series data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.logging_config import get_logger
from config.settings import settings
from core.exceptions import ModelError
from core.types import FeatureFrame, OHLCVFrame
from data.schema import SCHEMA
from data.validator import validate_ohlcv_frame

logger = get_logger(__name__)


@dataclass(slots=True)
class Dataset:
    """A feature matrix and label vector aligned on a shared index."""

    X: FeatureFrame
    y: pd.Series

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ModelError(
                f"X has {len(self.X)} rows but y has {len(self.y)}; they must match."
            )
        if not self.X.index.equals(self.y.index):
            raise ModelError("X and y must share the same index.")
        if self.X.empty:
            raise ModelError("Cannot build a Dataset from an empty feature frame.")

    @property
    def feature_names(self) -> list[str]:
        """The feature column names, in order."""
        return list(self.X.columns)

    def __len__(self) -> int:
        return len(self.X)


def make_forward_return_labels(
    frame: OHLCVFrame,
    horizon: int = 1,
    threshold: float = 0.0,
    *,
    column: str = SCHEMA.close,
) -> pd.Series:
    """
    Label each bar by its forward return over ``horizon`` bars.

    Parameters
    ----------
    frame : core.types.OHLCVFrame
        OHLCV bars conforming to :data:`data.schema.SCHEMA`.
    horizon : int, optional
        Number of bars ahead to measure the return over. Defaults to 1.
    threshold : float, optional
        Minimum absolute forward return required to label a bar ``1``
        (up) or ``-1`` (down); smaller moves are labeled ``0`` (flat).
        Defaults to ``0.0`` (pure binary sign; exact ties are ``0``).
    column : str, optional
        Price column to compute returns from. Defaults to
        :data:`data.schema.SCHEMA.close`.

    Returns
    -------
    pandas.Series
        Float labels in ``{-1.0, 0.0, 1.0}`` aligned to ``frame.index``.
        The last ``horizon`` bars are ``NaN`` (no future data to label
        them).

    Raises
    ------
    core.exceptions.ModelError
        If ``horizon`` is not positive or ``threshold`` is negative.
    """
    validate_ohlcv_frame(frame)

    if horizon <= 0:
        raise ModelError(f"horizon must be positive, got {horizon}.")
    if threshold < 0:
        raise ModelError(f"threshold must be non-negative, got {threshold}.")

    forward_return = frame[column].shift(-horizon) / frame[column] - 1.0

    labels = pd.Series(0.0, index=frame.index, dtype="float64")
    labels[forward_return > threshold] = 1.0
    labels[forward_return < -threshold] = -1.0
    labels[forward_return.isna()] = float("nan")

    return labels.rename("label")


def build_dataset(features: FeatureFrame, labels: pd.Series) -> Dataset:
    """
    Build a :class:`Dataset` from ``features`` and ``labels``, dropping unlabeled rows.

    Parameters
    ----------
    features : core.types.FeatureFrame
        Engineered feature columns, one row per bar.
    labels : pandas.Series
        Labels aligned to ``features.index`` (e.g. from
        :func:`make_forward_return_labels`), possibly containing NaNs.

    Returns
    -------
    Dataset
        The feature/label pair with any row containing a NaN (in
        either features or labels) removed.

    Raises
    ------
    core.exceptions.ModelError
        If ``features`` and ``labels`` do not share an index, or no
        rows remain after dropping NaNs.
    """
    if not features.index.equals(labels.index):
        raise ModelError("features and labels must share the same index.")

    valid = features.notna().all(axis=1) & labels.notna()
    clean_features = features.loc[valid]
    clean_labels = labels.loc[valid].astype("int64")

    if clean_features.empty:
        raise ModelError("No rows remain after dropping NaN features/labels.")

    logger.info(
        "Built dataset: %d/%d rows retained after dropping NaNs.",
        len(clean_features),
        len(features),
    )

    return Dataset(clean_features, clean_labels)


@dataclass(slots=True)
class ChronologicalSplit:
    """Train/validation/test partition of a :class:`Dataset`, split in time order."""

    train: Dataset
    validation: Dataset
    test: Dataset


def train_validation_test_split(
    dataset: Dataset,
    *,
    train_fraction: float | None = None,
    validation_fraction: float | None = None,
) -> ChronologicalSplit:
    """
    Split ``dataset`` chronologically into train/validation/test partitions.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split. Must already be sorted in time order (as
        any :class:`Dataset` built from an OHLCV-derived index is).
    train_fraction, validation_fraction : float | None, optional
        Fractions of rows to allocate to the train and validation
        partitions; the remainder goes to test. Default to
        :data:`config.settings.settings.ml.train_fraction` /
        ``.validation_fraction``.

    Returns
    -------
    ChronologicalSplit
        The train/validation/test partitions.

    Raises
    ------
    core.exceptions.ModelError
        If the fractions are not in ``(0, 1)``, don't leave room for a
        test set, or ``dataset`` is too small to split into three
        non-empty partitions.
    """
    train_fraction = (
        train_fraction if train_fraction is not None else settings.ml.train_fraction
    )
    validation_fraction = (
        validation_fraction
        if validation_fraction is not None
        else settings.ml.validation_fraction
    )

    if not (0 < train_fraction < 1) or not (0 < validation_fraction < 1):
        raise ModelError(
            "train_fraction and validation_fraction must both be in (0, 1)."
        )
    if train_fraction + validation_fraction >= 1:
        raise ModelError(
            "train_fraction + validation_fraction must leave room for a test set (< 1)."
        )

    n = len(dataset)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)

    if train_end == 0 or validation_end == train_end or validation_end >= n:
        raise ModelError(
            f"Dataset has {n} rows, too few to split into three non-empty "
            f"partitions with train_fraction={train_fraction}, "
            f"validation_fraction={validation_fraction}."
        )

    return ChronologicalSplit(
        train=Dataset(dataset.X.iloc[:train_end], dataset.y.iloc[:train_end]),
        validation=Dataset(
            dataset.X.iloc[train_end:validation_end],
            dataset.y.iloc[train_end:validation_end],
        ),
        test=Dataset(dataset.X.iloc[validation_end:], dataset.y.iloc[validation_end:]),
    )
