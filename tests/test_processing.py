"""Tests for src.processing module."""

import pandas as pd
import pytest

from src.processing import align_to_monthly


class TestAlignToMonthly:
    def test_daily_to_monthly(self):
        # 60 business days ~ 3 months
        idx = pd.bdate_range("2023-01-01", periods=60)
        df = pd.DataFrame({"val": range(60)}, index=idx)
        result = align_to_monthly(df, "daily")

        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3  # Jan, Feb, Mar
        assert result.index.freqstr == "MS"

    def test_quarterly_ffill(self):
        # Simulate PNAD: values at months 3, 6, 9, 12
        idx = pd.to_datetime(["2023-03-01", "2023-06-01", "2023-09-01", "2023-12-01"])
        df = pd.DataFrame({"desocupacao": [8.0, 8.5, 9.0, 8.8]}, index=idx)
        df.index.name = "date"

        result = align_to_monthly(df, "quarterly")

        # Should expand to 10 months (Mar-Dec)
        assert len(result) == 10
        # Apr and May should be forward-filled from Mar
        assert result.loc["2023-04-01", "desocupacao"] == 8.0
        assert result.loc["2023-05-01", "desocupacao"] == 8.0
        # Jul should be forward-filled from Jun
        assert result.loc["2023-07-01", "desocupacao"] == 8.5

    def test_monthly_passthrough(self):
        idx = pd.date_range("2023-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"selic": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}, index=idx)
        result = align_to_monthly(df, "monthly")

        assert len(result) == 6
        assert list(result["selic"]) == [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
