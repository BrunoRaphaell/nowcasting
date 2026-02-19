"""Align frequencies and merge raw series into a monthly panel."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "series.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def align_to_monthly(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Align a single-column DataFrame to monthly frequency.

    Args:
        df: DataFrame with DatetimeIndex and one column.
        frequency: One of 'monthly', 'daily', 'quarterly'.
    """
    col = df.columns[0]

    if frequency == "daily":
        # Resample daily -> monthly mean
        df = df.resample("MS").mean()
        logger.info("  %s: daily -> monthly mean (%d rows)", col, len(df))
    elif frequency == "quarterly":
        # PNAD trimestral móvel: forward-fill to monthly
        # The dates are already monthly (last month of the moving quarter)
        # Reindex to full monthly range and forward-fill
        full_range = pd.date_range(df.index.min(), df.index.max(), freq="MS")
        df = df.reindex(full_range).ffill()
        df.index.name = "date"
        logger.info("  %s: quarterly -> monthly ffill (%d rows)", col, len(df))
    else:
        logger.info("  %s: already monthly (%d rows)", col, len(df))

    return df


def build_panel(
    config_path: Path = CONFIG_PATH,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Load raw series, align to monthly, merge into a single panel DataFrame."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    all_series = config.get("targets", []) + config.get("features", [])

    frames = []
    for series in all_series:
        name = series["name"]
        frequency = series.get("frequency", "monthly")
        parquet_path = raw_dir / f"{name}.parquet"

        if not parquet_path.exists():
            logger.warning("Raw file not found: %s, skipping", parquet_path)
            continue

        df = pd.read_parquet(parquet_path)
        df = align_to_monthly(df, frequency)
        frames.append(df)

    if not frames:
        logger.error("No series loaded — cannot build panel")
        return pd.DataFrame()

    # Merge all on date index (outer join to keep full range)
    panel = frames[0]
    for df in frames[1:]:
        panel = panel.join(df, how="outer")

    panel = panel.sort_index()
    panel.index.name = "date"

    out_path = processed_dir / "panel.parquet"
    panel.to_parquet(out_path)
    logger.info(
        "Panel saved to %s: %d rows x %d cols, range %s to %s",
        out_path,
        len(panel),
        len(panel.columns),
        panel.index.min().date(),
        panel.index.max().date(),
    )
    return panel


if __name__ == "__main__":
    import src  # noqa: F401

    panel = build_panel()
    print(panel.head(10))
    print(f"\nShape: {panel.shape}")
    print(f"\nNull counts:\n{panel.isnull().sum()}")
