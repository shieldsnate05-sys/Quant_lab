"""
Quant-Lab Feature Selection.

Reduces a feature matrix down to a useful subset: low-variance and
highly-correlated columns are dropped structurally; importance-based
selection ranks the remainder using a fitted model's importances.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from config.logging_config import get_logger
from core.exceptions import ModelError
from core.types import FeatureFrame

logger = get_logger(__name__)


def drop_low_variance_features(
    features: FeatureFrame, threshold: float = 0.0
) -> FeatureFrame:
    """
    Drop columns whose variance is at or below ``threshold``.

    Parameters
    ----------
    features : core.types.FeatureFrame
        The feature matrix to prune.
    threshold : float, optional
        Minimum variance a column must exceed to be kept. Defaults to
        ``0.0`` (drops only exactly-constant columns). Financial
        return-scale features often have variance in the ``1e-4`` to
        ``1e-6`` range, so a non-zero default here would silently drop
        informative features - callers who want stricter pruning
        should pass an explicit ``threshold`` sized to their feature
        scale.

    Returns
    -------
    core.types.FeatureFrame
        ``features`` with low-variance columns removed.

    Raises
    ------
    core.exceptions.ModelError
        If every column is dropped.
    """
    selector = VarianceThreshold(threshold=threshold)

    try:
        selector.fit(features)
    except ValueError as exc:
        # sklearn raises (rather than returning empty support) when every
        # feature's variance is at or below the threshold.
        raise ModelError(
            f"All {features.shape[1]} features had variance <= {threshold}."
        ) from exc

    kept = features.columns[selector.get_support()]

    if len(kept) == 0:
        raise ModelError(
            f"All {features.shape[1]} features had variance <= {threshold}."
        )

    dropped = set(features.columns) - set(kept)
    if dropped:
        logger.info(
            "Dropped %d low-variance features: %s", len(dropped), sorted(dropped)
        )

    return features[list(kept)]


def drop_highly_correlated_features(
    features: FeatureFrame, threshold: float = 0.95
) -> FeatureFrame:
    """
    Drop columns that are highly correlated with an already-kept column.

    Iterates columns in order; a column is dropped if its absolute
    Pearson correlation with any previously-kept column exceeds
    ``threshold``.

    Parameters
    ----------
    features : core.types.FeatureFrame
        The feature matrix to prune.
    threshold : float, optional
        Maximum allowed absolute correlation. Defaults to ``0.95``.

    Returns
    -------
    core.types.FeatureFrame
        ``features`` with redundant columns removed.
    """
    correlation = features.corr().abs().to_numpy()
    columns = list(features.columns)
    kept: list[str] = []
    kept_positions: list[int] = []

    for position, column in enumerate(columns):
        if all(
            correlation[position, kept_pos] <= threshold for kept_pos in kept_positions
        ):
            kept.append(column)
            kept_positions.append(position)

    dropped = set(features.columns) - set(kept)
    if dropped:
        logger.info(
            "Dropped %d highly-correlated features: %s", len(dropped), sorted(dropped)
        )

    return features[kept]


def select_top_k_by_importance(
    features: FeatureFrame, importances: pd.Series, k: int
) -> FeatureFrame:
    """
    Keep only the ``k`` columns with the highest importance score.

    Parameters
    ----------
    features : core.types.FeatureFrame
        The feature matrix to prune.
    importances : pandas.Series
        Importance scores indexed by feature name (e.g. from
        :meth:`ml.base_model.BaseModel.feature_importances`).
    k : int
        Number of top features to keep.

    Returns
    -------
    core.types.FeatureFrame
        ``features`` restricted to the top ``k`` columns by importance.

    Raises
    ------
    core.exceptions.ModelError
        If ``k`` is not positive, or ``importances`` doesn't cover
        every column in ``features``.
    """
    if k <= 0:
        raise ModelError(f"k must be positive, got {k}.")

    missing = set(features.columns) - set(importances.index)
    if missing:
        raise ModelError(
            f"importances is missing scores for columns: {sorted(missing)}"
        )

    top_k = importances.loc[list(features.columns)].sort_values(ascending=False).head(k)

    return features[list(top_k.index)]
