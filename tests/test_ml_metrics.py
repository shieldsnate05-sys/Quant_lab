"""Tests for ml.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.exceptions import ModelError
from ml.metrics import compute_classification_metrics


def test_perfect_predictions_score_accuracy_one() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    y_proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.n_samples == 5


def test_metrics_without_proba_leave_binary_only_fields_none() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = compute_classification_metrics(y_true, y_pred, None)

    assert metrics.roc_auc is None
    assert metrics.log_loss is None
    assert metrics.brier_score is None
    assert 0 <= metrics.accuracy <= 1


def test_metrics_multiclass_skips_binary_only_fields() -> None:
    y_true = pd.Series([-1, 0, 1, -1, 0, 1])
    y_pred = np.array([-1, 0, 1, -1, 0, 1])
    y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics.roc_auc is None
    assert metrics.accuracy == pytest.approx(1.0)


def test_compute_metrics_rejects_empty_y_true() -> None:
    with pytest.raises(ModelError, match="empty"):
        compute_classification_metrics(pd.Series([], dtype="int64"), np.array([]))


def test_compute_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ModelError, match="rows"):
        compute_classification_metrics(pd.Series([0, 1]), np.array([0]))
