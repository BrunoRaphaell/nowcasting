"""Fetch raw time series from BCB/SGS and IBGE/SIDRA APIs."""

import logging
from pathlib import Path

import pandas as pd
import yaml
from bcb import sgs
import sidrapy

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "series.yaml"
RAW_DIR = ROOT / "data" / "raw"

# BCB limits daily series queries to 10 years
_BCB_DAILY_MAX_YEARS = 10


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load series configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_bcb_series(
    code: int, name: str, start_date: str = "2011-01-01", frequency: str = "monthly"
) -> pd.DataFrame:
    """Fetch a single series from BCB/SGS.

    For daily series, fetches in 10-year chunks to respect BCB API limits.
    Returns a DataFrame with DatetimeIndex and a single column named `name`.
    """
    logger.info("Fetching BCB series %s (code=%d) from %s", name, code, start_date)

    if frequency == "daily":
        # Fetch in chunks of 10 years
        chunks = []
        start = pd.Timestamp(start_date)
        end = pd.Timestamp.now()
        while start < end:
            chunk_end = min(start + pd.DateOffset(years=_BCB_DAILY_MAX_YEARS) - pd.DateOffset(days=1), end)
            logger.info("  chunk %s to %s", start.date(), chunk_end.date())
            chunk = sgs.get((name, code), start=str(start.date()), end=str(chunk_end.date()))
            chunks.append(chunk)
            start = chunk_end + pd.DateOffset(days=1)
        df = pd.concat(chunks) if chunks else pd.DataFrame()
    else:
        df = sgs.get((name, code), start=start_date)

    if not df.empty:
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="last")]
    logger.info("  -> %d rows fetched", len(df))
    return df


def fetch_ibge_series(
    table: int, variable: int, name: str
) -> pd.DataFrame:
    """Fetch a series from IBGE/SIDRA.

    Returns a DataFrame with DatetimeIndex and a single column named `name`.
    """
    logger.info("Fetching IBGE series %s (table=%d, var=%d)", name, table, variable)
    raw = sidrapy.get_table(
        table_code=str(table),
        territorial_level="1",
        ibge_territorial_code="1",
        variable=str(variable),
        period="all",
        header="n",
        format="list",
    )

    records = []
    for row in raw:
        period = row.get("D2C", "")  # period code like "202301"
        value = row.get("V", "")
        if not period or not value or value in ("...", "-", "X"):
            continue
        # Skip rows where D2C is the variable code (header artifact)
        if period == str(variable):
            continue
        try:
            val = float(value)
        except ValueError:
            continue
        records.append({"period": period, name: val})

    if not records:
        logger.warning("  -> No valid records for %s", name)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Period codes from PNAD Contínua trimestral móvel are like "202301"
    # meaning the trimester ending in January 2023. Parse as month.
    df["date"] = pd.to_datetime(df["period"], format="%Y%m")
    df = df.set_index("date").drop(columns=["period"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info("  -> %d rows fetched", len(df))
    return df


def ingest_all(
    config_path: Path = CONFIG_PATH,
    raw_dir: Path = RAW_DIR,
    start_date: str = "2011-01-01",
) -> None:
    """Fetch all series defined in config and save to raw_dir as parquet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    all_series = config.get("targets", []) + config.get("features", [])

    for series in all_series:
        name = series["name"]
        source = series["source"]
        frequency = series.get("frequency", "monthly")
        out_path = raw_dir / f"{name}.parquet"

        try:
            if source == "bcb":
                df = fetch_bcb_series(
                    code=series["code"],
                    name=name,
                    start_date=start_date,
                    frequency=frequency,
                )
            elif source == "ibge":
                df = fetch_ibge_series(
                    table=series["table"],
                    variable=series["variable"],
                    name=name,
                )
            else:
                logger.warning("Unknown source '%s' for series '%s', skipping", source, name)
                continue

            if df.empty:
                logger.warning("Empty result for %s, skipping save", name)
                continue

            df.to_parquet(out_path)
            logger.info("Saved %s -> %s", name, out_path)

        except Exception:
            logger.exception("Failed to fetch series '%s'", name)


if __name__ == "__main__":
    import src  # noqa: F401 — trigger logging config

    ingest_all()
