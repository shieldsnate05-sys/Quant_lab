"""
Quant-Lab ML Pipelines.

Wires the individual :mod:`ml` building blocks (preprocessing,
training, calibration, evaluation, persistence) into a single
end-to-end :class:`TrainingPipeline`, so a full model lifecycle can be
run in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.logging_config import get_logger
from ml.base_model import BaseModel
from ml.config import CalibrationConfig
from ml.dataset import ChronologicalSplit, Dataset
from ml.evaluate import EvaluationReport, evaluate_model
from ml.persistence import save_model
from ml.preprocessing import Preprocessor
from ml.train import calibrate_model, train_model

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """Everything produced by running a :class:`TrainingPipeline`."""

    model: BaseModel
    preprocessor: Preprocessor
    evaluation: EvaluationReport
    saved_path: Path | None


@dataclass(slots=True)
class TrainingPipeline:
    """
    An end-to-end train -> (calibrate) -> evaluate -> (persist) pipeline.

    Parameters
    ----------
    model : ml.base_model.BaseModel
        The (unfitted) model to train.
    preprocessor : ml.preprocessing.Preprocessor, optional
        Feature preprocessing, fit on the training split and applied
        to validation/test. Defaults to a new ``Preprocessor()``.
    calibrate : bool, optional
        Whether to calibrate predicted probabilities on the validation
        split after training. Defaults to ``False``.
    calibration_config : ml.config.CalibrationConfig | None, optional
        Calibration settings, used only if ``calibrate`` is ``True``.
    """

    model: BaseModel
    preprocessor: Preprocessor = field(default_factory=Preprocessor)
    calibrate: bool = False
    calibration_config: CalibrationConfig | None = None

    def run(
        self, split: ChronologicalSplit, *, model_version: str | None = None
    ) -> PipelineResult:
        """
        Run the full pipeline over ``split``.

        Parameters
        ----------
        split : ml.dataset.ChronologicalSplit
            Train/validation/test partitions.
        model_version : str | None, optional
            If given, the trained (and possibly calibrated) model is
            persisted under this version via :func:`ml.persistence.save_model`.

        Returns
        -------
        PipelineResult
        """
        train_X = self.preprocessor.fit_transform(split.train.X)
        train_dataset = Dataset(train_X, split.train.y)

        model = train_model(self.model, train_dataset)

        if self.calibrate:
            validation_X = self.preprocessor.transform(split.validation.X)
            validation_dataset = Dataset(validation_X, split.validation.y)
            model = calibrate_model(model, validation_dataset, self.calibration_config)

        test_X = self.preprocessor.transform(split.test.X)
        test_dataset = Dataset(test_X, split.test.y)
        evaluation = evaluate_model(model, test_dataset)

        saved_path = None
        if model_version is not None:
            saved_path = save_model(
                model, model_version, feature_names=train_dataset.feature_names
            )

        logger.info(
            "Pipeline complete for %s: test accuracy=%.4f f1=%.4f",
            model.name,
            evaluation.metrics.accuracy,
            evaluation.metrics.f1,
        )

        return PipelineResult(
            model=model,
            preprocessor=self.preprocessor,
            evaluation=evaluation,
            saved_path=saved_path,
        )
