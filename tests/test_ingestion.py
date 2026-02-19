"""Tests for src.ingestion module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.ingestion import fetch_bcb_series, fetch_ibge_series, ingest_all, load_config


# --- Fixtures ---

@pytest.fixture
def sample_config(tmp_path):
    config = tmp_path / "series.yaml"
    config.write_text(
        """
targets:
  - name: test_target
    source: bcb
    code: 21084
    frequency: monthly

features:
  - name: test_bcb_feature
    source: bcb
    code: 4390
    frequency: monthly

  - name: test_ibge_feature
    source: ibge
    table: 6381
    variable: 4099
    frequency: quarterly
"""
    )
    return config


@pytest.fixture
def mock_bcb_df():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    return pd.DataFrame({"test_col": range(12)}, index=idx)


@pytest.fixture
def mock_ibge_response():
    return [
        {"D2C": "202001", "V": "11.2"},
        {"D2C": "202002", "V": "11.5"},
        {"D2C": "202003", "V": "11.8"},
        {"D2C": "202004", "V": "..."},  # missing value
        {"D2C": "202005", "V": "12.1"},
    ]


# --- Unit tests ---

class TestLoadConfig:
    def test_loads_yaml(self, sample_config):
        config = load_config(sample_config)
        assert "targets" in config
        assert "features" in config
        assert len(config["targets"]) == 1
        assert config["targets"][0]["code"] == 21084


class TestFetchBCB:
    @patch("src.ingestion.sgs.get")
    def test_returns_dataframe(self, mock_get, mock_bcb_df):
        mock_get.return_value = mock_bcb_df
        result = fetch_bcb_series(code=21084, name="test", start_date="2020-01-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12
        assert result.index.name == "date"
        mock_get.assert_called_once_with(("test", 21084), start="2020-01-01")


class TestFetchIBGE:
    @patch("src.ingestion.sidrapy.get_table")
    def test_returns_dataframe(self, mock_get_table, mock_ibge_response):
        mock_get_table.return_value = mock_ibge_response
        result = fetch_ibge_series(table=6381, variable=4099, name="desocupacao")
        assert isinstance(result, pd.DataFrame)
        assert "desocupacao" in result.columns
        # Should have 4 valid rows (one "..." skipped)
        assert len(result) == 4

    @patch("src.ingestion.sidrapy.get_table")
    def test_empty_response(self, mock_get_table):
        mock_get_table.return_value = []
        result = fetch_ibge_series(table=6381, variable=4099, name="test")
        assert result.empty


class TestIngestAll:
    @patch("src.ingestion.fetch_ibge_series")
    @patch("src.ingestion.fetch_bcb_series")
    def test_saves_parquet_files(self, mock_bcb, mock_ibge, sample_config, tmp_path, mock_bcb_df):
        raw_dir = tmp_path / "raw"
        mock_bcb.return_value = mock_bcb_df
        ibge_df = pd.DataFrame(
            {"test_ibge_feature": [10.0, 11.0]},
            index=pd.to_datetime(["2020-01-01", "2020-04-01"]),
        )
        mock_ibge.return_value = ibge_df

        ingest_all(config_path=sample_config, raw_dir=raw_dir)

        assert (raw_dir / "test_target.parquet").exists()
        assert (raw_dir / "test_bcb_feature.parquet").exists()
        assert (raw_dir / "test_ibge_feature.parquet").exists()


# --- Integration tests ---

@pytest.mark.slow
class TestIntegrationBCB:
    def test_fetch_real_selic(self):
        df = fetch_bcb_series(code=4390, name="selic", start_date="2023-01-01")
        assert not df.empty
        assert "selic" in df.columns


@pytest.mark.slow
class TestIntegrationIBGE:
    def test_fetch_real_desocupacao(self):
        df = fetch_ibge_series(table=6381, variable=4099, name="desocupacao")
        assert not df.empty
        assert "desocupacao" in df.columns
