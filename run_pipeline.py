#!/usr/bin/env python
"""Run the full nowcasting pipeline: ingestion -> processing -> features -> evaluation."""

import logging
import warnings

import src  # noqa: F401 — trigger logging config

from src.ingestion import ingest_all
from src.processing import build_panel
from src.features import build_features
from src.evaluation import (
    walk_forward_evaluate,
    save_results,
    plot_predictions,
    plot_model_comparison,
    plot_feature_importance,
)
from src.models import get_all_models

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Stage 1: Ingestion ===")
    ingest_all()

    logger.info("=== Stage 2: Processing ===")
    panel = build_panel()

    logger.info("=== Stage 3: Feature Engineering ===")
    X, y = build_features(panel)

    logger.info("=== Stage 4+5: Modeling & Evaluation ===")
    models = get_all_models()
    pred_df, metrics_df = walk_forward_evaluate(X, y, models=models)
    save_results(pred_df, metrics_df)

    logger.info("=== Stage 6: Visualization ===")
    plot_predictions(pred_df)
    plot_model_comparison(metrics_df)
    plot_feature_importance(models)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
