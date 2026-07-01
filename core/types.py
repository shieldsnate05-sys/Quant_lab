"""
Quant-Lab Core Type Aliases.

Common type aliases shared across the platform so that function
signatures stay both readable and precisely typed. These are plain
aliases (PEP 695 ``type`` statements), not generic type variables -
use them to annotate parameters and return values, not to
parameterize generic classes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

#: A ticker symbol, e.g. ``"QQQ"``.
type Symbol = str

#: A DataFrame with an OHLCV schema (open, high, low, close, volume).
type OHLCVFrame = pd.DataFrame

#: A single price series, e.g. a close column.
type PriceSeries = pd.Series

#: A single volume series.
type VolumeSeries = pd.Series

#: A DataFrame of engineered features, one column per feature.
type FeatureFrame = pd.DataFrame

#: A dense array of model predictions.
type PredictionArray = npt.NDArray[np.float64]
