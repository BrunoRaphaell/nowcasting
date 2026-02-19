"""Model definitions with a consistent fit/predict interface."""

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Common interface for all nowcasting models."""

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class PersistenceModel(BaseModel):
    """Naive baseline: predict y_{t-1}."""

    name = "Persistence"

    def __init__(self):
        self._last_value = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._last_value = y.iloc[-1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._last_value)


class ARIMAModel(BaseModel):
    """Univariate ARIMA with auto order selection via AIC."""

    name = "ARIMA"

    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self._model_fit = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_clean = y.dropna()
        best_aic = np.inf
        best_order = (1, 0, 0)

        for d in range(self.max_d + 1):
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(y_clean, order=(p, d, q))
                            fit = model.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        logger.info("ARIMA best order: %s (AIC=%.2f)", best_order, best_aic)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_fit = ARIMA(y_clean, order=best_order).fit()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        forecast = self._model_fit.forecast(steps=n)
        return np.asarray(forecast)


# ---------------------------------------------------------------------------
# Macro-augmented models
# ---------------------------------------------------------------------------

class SARIMAXModel(BaseModel):
    """SARIMAX with exogenous features. Falls back to fewer features on convergence failure."""

    name = "SARIMAX"

    def __init__(self, max_p: int = 3, max_d: int = 1, max_q: int = 3):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self._model_fit = None
        self._used_cols: list[str] = []

    def _fit_with_exog(self, y: pd.Series, exog: pd.DataFrame) -> bool:
        """Try to fit SARIMAX with given exogenous vars. Returns True on success."""
        # Drop rows with NaN in either y or exog
        mask = y.notna() & exog.notna().all(axis=1)
        y_clean = y[mask]
        exog_clean = exog[mask]

        if len(y_clean) < 36:
            logger.warning("SARIMAX: not enough clean observations (%d)", len(y_clean))
            return False

        best_aic = np.inf
        best_order = (1, 0, 0)

        for d in range(self.max_d + 1):
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = SARIMAX(
                                y_clean, exog=exog_clean, order=(p, d, q),
                                enforce_stationarity=False, enforce_invertibility=False,
                            )
                            fit = model.fit(disp=False, maxiter=200)
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        if best_aic == np.inf:
            return False

        logger.info("SARIMAX best order: %s (AIC=%.2f) with %d exog vars", best_order, best_aic, exog_clean.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_clean, exog=exog_clean, order=best_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self._model_fit = model.fit(disp=False, maxiter=200)
        self._used_cols = list(exog_clean.columns)
        return True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Try with all features first
        if self._fit_with_exog(y, X):
            return

        # Fallback: use only lag-1 columns
        lag1_cols = [c for c in X.columns if c.endswith("_lag1")]
        logger.warning("SARIMAX: falling back to lag-1 only (%d vars)", len(lag1_cols))
        if lag1_cols and self._fit_with_exog(y, X[lag1_cols]):
            return

        # Last resort: fit without exogenous
        logger.warning("SARIMAX: fitting without exogenous variables")
        y_clean = y.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y_clean, order=(1, 0, 0))
            self._model_fit = model.fit(disp=False)
        self._used_cols = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if self._used_cols:
            exog_future = X[self._used_cols].iloc[:n]
            # Fill NaN with last known value for prediction
            exog_future = exog_future.ffill().bfill()
            forecast = self._model_fit.forecast(steps=n, exog=exog_future)
        else:
            forecast = self._model_fit.forecast(steps=n)
        return np.asarray(forecast)


class ElasticNetModel(BaseModel):
    """Elastic Net with built-in CV for alpha and l1_ratio. Scales features."""

    name = "ElasticNet"

    def __init__(self):
        self._scaler = StandardScaler()
        self._model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            cv=TimeSeriesSplit(n_splits=5),
            max_iter=10000,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        mask = y.notna() & X.notna().all(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask].values

        self._feature_names = list(X.columns)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_clean)
        self._model.fit(X_scaled, y_clean)
        logger.info(
            "ElasticNet: alpha=%.4f, l1_ratio=%.2f",
            self._model.alpha_, self._model.l1_ratio_,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Fill NaN for prediction (use column means from training)
        X_filled = X.fillna(X.mean())
        X_scaled = self._scaler.transform(X_filled.values)
        return self._model.predict(X_scaled)


class XGBoostModel(BaseModel):
    """XGBoost with nested TimeSeriesSplit CV for hyperparameters."""

    name = "XGBoost"

    def __init__(self):
        self._model: XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # XGBoost handles NaN natively, but target must not be NaN
        mask = y.notna()
        X_train = X[mask]
        y_train = y[mask]

        # Nested CV for hyperparameter selection
        best_score = np.inf
        best_params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}

        param_grid = [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [50, 100, 200]
            for d in [3, 5]
            for lr in [0.05, 0.1]
        ]

        tscv = TimeSeriesSplit(n_splits=3)

        for params in param_grid:
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                model = XGBRegressor(
                    **params,
                    random_state=42,
                    verbosity=0,
                    enable_categorical=False,
                )
                model.fit(
                    X_train.iloc[train_idx], y_train.iloc[train_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    verbose=False,
                )
                preds = model.predict(X_train.iloc[val_idx])
                mse = np.mean((y_train.iloc[val_idx].values - preds) ** 2)
                scores.append(mse)

            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        logger.info("XGBoost best params: %s (CV MSE=%.4f)", best_params, best_score)

        self._model = XGBRegressor(
            **best_params,
            random_state=42,
            verbosity=0,
            enable_categorical=False,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(self._model.feature_names_in_, self._model.feature_importances_))


def get_all_models() -> list[BaseModel]:
    """Return instances of all models to evaluate."""
    return [
        PersistenceModel(),
        ARIMAModel(),
        SARIMAXModel(),
        ElasticNetModel(),
        XGBoostModel(),
    ]
