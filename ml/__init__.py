"""
Quant-Lab Machine Learning Package.

Classification models for direction/label prediction, built around a
common :class:`~ml.base_model.BaseModel` interface:

- :mod:`ml.dataset` - labeling and chronological train/validation/test splits
- :mod:`ml.preprocessing` - imputation and scaling
- :mod:`ml.feature_selection` - variance, correlation, and importance-based pruning
- :mod:`ml.random_forest` / :mod:`ml.xgboost_model` - concrete models,
  self-registering with :data:`~ml.model_registry.MODEL_REGISTRY`
- :mod:`ml.train` - fitting, hyperparameter tuning, probability calibration
- :mod:`ml.cross_validation` - ``TimeSeriesSplit``-based cross-validation
- :mod:`ml.walk_forward` - rolling/expanding walk-forward validation
- :mod:`ml.evaluate` / :mod:`ml.metrics` - scoring
- :mod:`ml.feature_importance` / :mod:`ml.shap_analysis` - explainability
- :mod:`ml.persistence` / :mod:`ml.predict` - model I/O and inference
- :mod:`ml.pipelines` - end-to-end orchestration
"""

from __future__ import annotations

from ml.base_model import BaseModel
from ml.config import (
    CalibrationConfig,
    RandomForestConfig,
    RandomForestSearchSpace,
    WalkForwardConfig,
    XGBoostConfig,
    XGBoostSearchSpace,
)
from ml.cross_validation import (
    CrossValidationResult,
    cross_validate_model,
    make_time_series_split,
)
from ml.dataset import (
    ChronologicalSplit,
    Dataset,
    build_dataset,
    make_forward_return_labels,
    train_validation_test_split,
)
from ml.evaluate import EvaluationReport, evaluate_model
from ml.feature_importance import built_in_importance, permutation_feature_importance
from ml.feature_selection import (
    drop_highly_correlated_features,
    drop_low_variance_features,
    select_top_k_by_importance,
)
from ml.metrics import ClassificationMetrics, compute_classification_metrics
from ml.model_registry import MODEL_REGISTRY, ModelRegistry
from ml.persistence import (
    ModelMetadata,
    list_versions,
    load_metadata,
    load_model,
    save_model,
)
from ml.pipelines import PipelineResult, TrainingPipeline
from ml.predict import predict, predict_proba, predict_with_confidence, save_predictions
from ml.preprocessing import Preprocessor
from ml.random_forest import RandomForestModel
from ml.shap_analysis import ShapResult, compute_shap_values
from ml.train import TuningResult, calibrate_model, train_model, tune_hyperparameters
from ml.walk_forward import WalkForwardResult, WalkForwardWindow, run_walk_forward
from ml.xgboost_model import XGBoostModel

__all__ = [
    "MODEL_REGISTRY",
    "BaseModel",
    "CalibrationConfig",
    "ChronologicalSplit",
    "ClassificationMetrics",
    "CrossValidationResult",
    "Dataset",
    "EvaluationReport",
    "ModelMetadata",
    "ModelRegistry",
    "PipelineResult",
    "Preprocessor",
    "RandomForestConfig",
    "RandomForestModel",
    "RandomForestSearchSpace",
    "ShapResult",
    "TrainingPipeline",
    "TuningResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    "XGBoostConfig",
    "XGBoostModel",
    "XGBoostSearchSpace",
    "build_dataset",
    "built_in_importance",
    "calibrate_model",
    "compute_classification_metrics",
    "compute_shap_values",
    "cross_validate_model",
    "drop_highly_correlated_features",
    "drop_low_variance_features",
    "evaluate_model",
    "list_versions",
    "load_metadata",
    "load_model",
    "make_forward_return_labels",
    "make_time_series_split",
    "permutation_feature_importance",
    "predict",
    "predict_proba",
    "predict_with_confidence",
    "run_walk_forward",
    "save_model",
    "save_predictions",
    "select_top_k_by_importance",
    "train_model",
    "train_validation_test_split",
    "tune_hyperparameters",
]
