"""Walk-forward evaluation and visualization."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models import BaseModel, get_all_models

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

MIN_TRAIN_WINDOW = 36


def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[BaseModel] | None = None,
    min_train: int = MIN_TRAIN_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward expanding-window evaluation.

    Args:
        X: Feature matrix (full, including NaN rows).
        y: Target series.
        models: List of model instances to evaluate.
        min_train: Minimum number of observations in the initial training window.

    Returns:
        predictions: DataFrame with columns [date, actual, model1, model2, ...]
        metrics: DataFrame with columns [model, MAE, RMSE, DirectionalAccuracy]
    """
    if models is None:
        models = get_all_models()

    # Only keep rows where target is not NaN
    valid_mask = y.notna()
    valid_dates = y[valid_mask].index
    n_valid = len(valid_dates)

    if n_valid <= min_train:
        raise ValueError(
            f"Not enough valid observations ({n_valid}) for min_train={min_train}"
        )

    n_test = n_valid - min_train
    logger.info(
        "Walk-forward: %d valid obs, %d train init, %d test steps",
        n_valid, min_train, n_test,
    )

    # Collect predictions
    results = {m.name: [] for m in models}
    actuals = []
    dates = []

    for step in range(n_test):
        train_end_idx = min_train + step
        test_idx = train_end_idx

        train_dates = valid_dates[:train_end_idx]
        test_date = valid_dates[test_idx]

        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]
        X_test = X.loc[[test_date]]
        y_actual = y.loc[test_date]

        actuals.append(y_actual)
        dates.append(test_date)

        if step % 20 == 0:
            logger.info("  step %d/%d (test date: %s)", step + 1, n_test, test_date.date())

        for model in models:
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                results[model.name].append(pred[0])
            except Exception:
                logger.exception("  %s failed at step %d", model.name, step)
                results[model.name].append(np.nan)

    # Build predictions DataFrame
    pred_df = pd.DataFrame({"date": dates, "actual": actuals})
    for model in models:
        pred_df[model.name] = results[model.name]
    pred_df = pred_df.set_index("date")

    # Compute metrics
    metrics_rows = []
    for model in models:
        preds = pred_df[model.name].values
        actual = pred_df["actual"].values
        valid = ~np.isnan(preds)

        if valid.sum() == 0:
            metrics_rows.append({"model": model.name, "MAE": np.nan, "RMSE": np.nan, "DirectionalAccuracy": np.nan})
            continue

        errors = actual[valid] - preds[valid]
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        # Directional accuracy: did we predict the right direction of change?
        actual_dir = np.diff(actual[valid])
        pred_dir = np.diff(preds[valid])
        if len(actual_dir) > 0:
            dir_acc = np.mean(np.sign(actual_dir) == np.sign(pred_dir))
        else:
            dir_acc = np.nan

        metrics_rows.append({
            "model": model.name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "DirectionalAccuracy": round(dir_acc, 4),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    return pred_df, metrics_df


def save_results(
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Save prediction and metrics DataFrames to CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / "predictions.csv"
    metrics_path = results_dir / "model_comparison.csv"

    pred_df.to_csv(pred_path)
    metrics_df.to_csv(metrics_path, index=False)

    logger.info("Predictions saved to %s", pred_path)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("\n%s", metrics_df.to_string(index=False))


def plot_predictions(
    pred_df: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Plot actual vs predicted for each model."""
    results_dir.mkdir(parents=True, exist_ok=True)
    model_cols = [c for c in pred_df.columns if c != "actual"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pred_df.index, pred_df["actual"], "k-", linewidth=2, label="Actual")
    for col in model_cols:
        ax.plot(pred_df.index, pred_df[col], "--", alpha=0.7, label=col)
    ax.set_title("Nowcast: Actual vs Predicted — Inadimplência PF Total")
    ax.set_ylabel("Inadimplência (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    logger.info("Saved actual_vs_predicted.png")


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Bar chart comparing model metrics."""
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "DirectionalAccuracy"]):
        sns.barplot(data=metrics_df, x="model", y=metric, ax=ax, hue="model", legend=False)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved model_comparison.png")


def plot_feature_importance(
    models: list[BaseModel],
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Plot feature importance from XGBoost and Elastic Net coefficients."""
    results_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        if model.name == "XGBoost" and hasattr(model, "feature_importances"):
            importances = model.feature_importances
            if importances is None:
                continue
            imp_df = (
                pd.Series(importances)
                .sort_values(ascending=True)
                .tail(15)
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            imp_df.plot.barh(ax=ax)
            ax.set_title("XGBoost — Top 15 Feature Importances")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fig.savefig(results_dir / "xgboost_importance.png", dpi=150)
            plt.close(fig)
            logger.info("Saved xgboost_importance.png")

        if model.name == "ElasticNet" and hasattr(model, "_model"):
            coefs = model._model.coef_
            feature_names = getattr(model, "_feature_names", None)
            if feature_names is None:
                continue
            coef_df = (
                pd.Series(coefs, index=feature_names)
                .abs()
                .sort_values(ascending=True)
                .tail(15)
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            coef_df.plot.barh(ax=ax)
            ax.set_title("Elastic Net — Top 15 |Coefficients|")
            ax.set_xlabel("|Coefficient|")
            plt.tight_layout()
            fig.savefig(results_dir / "elasticnet_coefficients.png", dpi=150)
            plt.close(fig)
            logger.info("Saved elasticnet_coefficients.png")
