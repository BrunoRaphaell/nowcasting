"""Tests for src.features module."""

import pandas as pd
import numpy as np

from src.features import build_features


def _make_panel(n: int = 24) -> pd.DataFrame:
    """Create a simple monthly panel for testing."""
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "inadimplencia_pf_total": rng.uniform(3, 6, n),
            "selic_acumulada_mes": rng.uniform(0.5, 1.5, n),
            "ibc_br_dessaz": rng.uniform(130, 145, n),
        },
        index=idx,
    )


class TestBuildFeatures:
    def test_output_shapes(self):
        panel = _make_panel()
        X, y = build_features(panel, feature_cols=["selic_acumulada_mes", "ibc_br_dessaz"])
        assert len(X) == len(panel)
        assert len(y) == len(panel)
        # 2 features x (3 lags + 2 deltas) = 10, plus 3 AR lags = 13
        assert X.shape[1] == 13

    def test_no_contemporaneous_leakage(self):
        """Features at time t should only use information from t-1 or earlier."""
        panel = _make_panel()
        X, y = build_features(panel, feature_cols=["selic_acumulada_mes"])

        # lag1 at index 1 should equal the original value at index 0
        selic_vals = panel["selic_acumulada_mes"].values
        lag1_vals = X["selic_acumulada_mes_lag1"].values

        # lag1[1] = selic[0], lag1[2] = selic[1], etc.
        for t in range(1, len(panel)):
            assert lag1_vals[t] == selic_vals[t - 1], f"Leakage at t={t}"

    def test_lag1_is_nan_at_first_row(self):
        panel = _make_panel()
        X, _ = build_features(panel, feature_cols=["selic_acumulada_mes"])
        assert pd.isna(X["selic_acumulada_mes_lag1"].iloc[0])

    def test_delta_computation(self):
        panel = _make_panel()
        X, _ = build_features(panel, feature_cols=["selic_acumulada_mes"])

        # delta1 = shift(1) - shift(2)
        selic = panel["selic_acumulada_mes"]
        expected_delta1 = selic.shift(1) - selic.shift(2)
        pd.testing.assert_series_equal(
            X["selic_acumulada_mes_delta1"],
            expected_delta1,
            check_names=False,
        )

    def test_ar_lags_present(self):
        panel = _make_panel()
        X, _ = build_features(panel, feature_cols=["selic_acumulada_mes"])
        ar_cols = [c for c in X.columns if "_ar_lag" in c]
        assert len(ar_cols) == 3  # AR lags 1, 3, 6
