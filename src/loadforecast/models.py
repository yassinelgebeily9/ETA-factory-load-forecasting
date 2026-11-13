"""Forecasting models. Each one maps a frame of origins to a (n, 24) forecast."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import HORIZONS, SEED, WINDOW
from .features import horizon_features


class Persistence:
    """y(t+h) = y(t)."""
    name = "Persistence"

    def fit(self, origin, course, Y):
        return self

    def predict(self, origin, course):
        return np.tile(origin["load_lag_0"].to_numpy()[:, None], (1, len(HORIZONS)))


class SeasonalNaive:
    """y(t+h) = y(t+h-24): the same hour yesterday."""
    name = "Seasonal naive (same hour yesterday)"

    def fit(self, origin, course, Y):
        return self

    def predict(self, origin, course):
        pred = np.column_stack([origin[f"load_lag_{WINDOW - h}"] for h in HORIZONS])
        # If that hour is missing in the window, fall back to the current load.
        fallback = np.tile(origin["load_lag_0"].to_numpy()[:, None], (1, len(HORIZONS)))
        return np.where(np.isnan(pred), fallback, pred)


class CourseGB:
    """The original group's final model: multi-output HistGB on its feature set."""
    name = "Course model (HistGB, original features)"

    def fit(self, origin, course, Y):
        base = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.075, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=0.1, random_state=SEED,
        )
        self.model = MultiOutputRegressor(base).fit(course, Y)
        return self

    def predict(self, origin, course):
        return self.model.predict(course)


class Direct:
    """One model per horizon on origin features + target-hour features."""

    def __init__(self, name, factory):
        self.name = name
        self.factory = factory

    @staticmethod
    def _X(origin, h):
        return pd.concat([origin, horizon_features(origin.index, origin, h)], axis=1)

    def fit(self, origin, course, Y):
        self.models = [self.factory().fit(self._X(origin, h), Y[:, h - 1]) for h in HORIZONS]
        return self

    def predict(self, origin, course):
        return np.column_stack(
            [m.predict(self._X(origin, h)) for h, m in zip(HORIZONS, self.models)]
        )


def ridge():
    return Direct(
        "Ridge (direct, new features)",
        lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
    )


def lightgbm():
    return Direct(
        "LightGBM (direct, new features)",
        lambda: LGBMRegressor(
            objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1,
        ),
    )


class Ensemble:
    """Plain average of other models' forecasts (members are named, not refitted)."""

    def __init__(self, name, member_names):
        self.name = name
        self.member_names = member_names


def all_models():
    return [
        Persistence(),
        SeasonalNaive(),
        CourseGB(),
        ridge(),
        lightgbm(),
        Ensemble("Ensemble (LightGBM + Ridge)", ["LightGBM (direct, new features)",
                                                 "Ridge (direct, new features)"]),
    ]
