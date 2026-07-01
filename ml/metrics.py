"""
Quant-Lab ML Classification Metrics.

Wraps scikit-learn's classification metrics in a single typed
:class:`ClassificationMetrics` result, computed consistently everywhere
a model is scored (cross-validation, walk-forward, evaluation reports).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.exceptions import ModelError


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Summary classification metrics for a single evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    log_loss: float | None
    brier_score: float | None
    n_samples: int


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    positive_class: int = 1,
) -> ClassificationMetrics:
    """
    Compute :class:`ClassificationMetrics` for a set of predictions.

    Parameters
    ----------
    y_true : pandas.Series
        Ground-truth labels.
    y_pred : numpy.ndarray
        Predicted labels, aligned to ``y_true``.
    y_proba : numpy.ndarray | None, optional
        Predicted class probabilities (as returned by
        :meth:`~ml.base_model.BaseModel.predict_proba`), shape
        ``(n_samples, n_classes)``. Required to compute ``roc_auc``,
        ``log_loss``, and ``brier_score``; those fields are ``None``
        if not provided or the problem is not binary.
    positive_class : int, optional
        The label treated as the positive class for ROC-AUC and Brier
        score (both binary-only metrics). Defaults to ``1``.

    Returns
    -------
    ClassificationMetrics
        The computed metrics. ``average="macro"`` is used for
        precision/recall/F1 so multi-class problems are scored evenly
        across classes.

    Raises
    ------
    core.exceptions.ModelError
        If ``y_true`` is empty or ``y_true``/``y_pred`` lengths differ.
    """
    if len(y_true) == 0:
        raise ModelError("Cannot compute metrics on empty y_true.")
    if len(y_true) != len(y_pred):
        raise ModelError(f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}.")

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    roc_auc: float | None = None
    log_loss_value: float | None = None
    brier: float | None = None

    classes = sorted(pd.unique(y_true))
    if y_proba is not None and len(classes) == 2 and positive_class in classes:
        positive_index = classes.index(positive_class)
        positive_proba = y_proba[:, positive_index]

        roc_auc = float(roc_auc_score(y_true, positive_proba))
        log_loss_value = float(log_loss(y_true, y_proba, labels=classes))
        brier = float(brier_score_loss(y_true == positive_class, positive_proba))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        log_loss=log_loss_value,
        brier_score=brier,
        n_samples=len(y_true),
    )
