"""Feature engineering: lags, deltas, and autoregressive terms."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET_COL = "inadimplencia_pf_total"
FEATURE_COLS = [
    "inadimplencia_carteira_total",
    "selic_acumulada_mes",
    "ibc_br_dessaz",
    "cambio_ptax_venda",
    "taxa_desocupacao",
    "rendimento_medio_real",
]

# Feature parameters (per interview decisions)
LAG_PERIODS = [1, 3, 6]
DELTA_PERIODS = [1, 3]
AR_LAGS = [1, 3, 6]  # autoregressive lags of the target


def build_features(
    panel: pd.DataFrame,
    target_col: str = TARGET_COL,
    feature_cols: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix from the monthly panel.

    All features use .shift(1) at minimum to prevent data leakage:
    - lag_k actually means shift(k) — value from k months ago
    - delta_k means the difference between shift(1) and shift(1+k)

    Returns:
        X: feature DataFrame (rows with NaN from lagging are NOT dropped)
        y: target Series aligned with X
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    features = {}

    # Exogenous variable features
    for col in feature_cols:
        if col not in panel.columns:
            logger.warning("Column '%s' not in panel, skipping", col)
            continue

        series = panel[col]

        # Lags (shift by at least 1 to avoid leakage)
        for lag in LAG_PERIODS:
            features[f"{col}_lag{lag}"] = series.shift(lag)

        # Deltas: change over the last k periods, observed at t-1
        for delta in DELTA_PERIODS:
            features[f"{col}_delta{delta}"] = series.shift(1) - series.shift(1 + delta)

    # Autoregressive lags of the target
    target = panel[target_col]
    for lag in AR_LAGS:
        features[f"{target_col}_ar_lag{lag}"] = target.shift(lag)

    X = pd.DataFrame(features, index=panel.index)
    y = target.copy()

    logger.info(
        "Feature matrix: %d rows x %d cols, target: %d non-null values",
        len(X),
        len(X.columns),
        y.notna().sum(),
    )
    return X, y


def load_panel_and_build_features(
    processed_dir: Path = PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load panel.parquet and build the feature matrix."""
    panel = pd.read_parquet(processed_dir / "panel.parquet")
    return build_features(panel)


if __name__ == "__main__":
    import src  # noqa: F401

    X, y = load_panel_and_build_features()
    print(f"X shape: {X.shape}")
    print(f"Feature columns:\n{list(X.columns)}")
    print(f"\ny non-null: {y.notna().sum()}")
    print(f"\nX null counts (top 10):\n{X.isnull().sum().nlargest(10)}")
