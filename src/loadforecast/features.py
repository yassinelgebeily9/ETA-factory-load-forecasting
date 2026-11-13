"""Feature engineering.

Every feature for forecast origin t uses only the 24 rows t-23 ... t (the same
history a test slice provides) plus calendar facts about the future, which
are known in advance (hour, weekday, public holidays).
"""
import numpy as np
import pandas as pd

from .config import HORIZONS, TARGET, WEATHER, WINDOW
from .data import is_holiday


def _offday(index: pd.DatetimeIndex) -> np.ndarray:
    hol = is_holiday(pd.Series(index.date, index=index)).to_numpy()
    return ((index.dayofweek >= 5) | hol).astype(int)


def origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features shared by all horizons, computed at each origin t."""
    y = df[TARGET]
    feats = {}

    # The full 24-hour load window: lag 0 = load at t, lag 23 = load at t-23.
    for lag in range(WINDOW):
        feats[f"load_lag_{lag}"] = y.shift(lag)

    roll = y.rolling(WINDOW, min_periods=WINDOW // 2)
    feats["load_mean_24"] = roll.mean()
    feats["load_std_24"] = roll.std()
    feats["load_min_24"] = roll.min()
    feats["load_max_24"] = roll.max()
    feats["load_mean_3"] = y.rolling(3, min_periods=1).mean()
    feats["load_diff_1"] = y - y.shift(1)

    for col in WEATHER:
        feats[col] = df[col]
        feats[f"{col}_mean_24"] = df[col].rolling(WINDOW, min_periods=1).mean()
    feats["temp_change_24"] = (
        df["temperature_air_mean_C"] - df["temperature_air_mean_C"].shift(WINDOW - 1)
    )

    idx = df.index
    feats["origin_hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feats["origin_hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feats["origin_offday"] = _offday(idx)
    feats["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return pd.DataFrame(feats, index=idx)


def horizon_features(index: pd.DatetimeIndex, origin: pd.DataFrame, h: int) -> pd.DataFrame:
    """Extra features describing the target hour t+h."""
    target_time = index + pd.Timedelta(hours=h)
    return pd.DataFrame(
        {
            "target_hour_sin": np.sin(2 * np.pi * target_time.hour / 24),
            "target_hour_cos": np.cos(2 * np.pi * target_time.hour / 24),
            "target_dow": target_time.dayofweek,
            "target_offday": _offday(target_time),
            # Same hour yesterday: y(t+h-24), always inside the 24h window.
            "same_hour_yesterday": origin[f"load_lag_{WINDOW - h}"].to_numpy(),
        },
        index=index,
    )


def targets(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET]
    return pd.DataFrame({f"h{h}": y.shift(-h) for h in HORIZONS}, index=df.index)


def course_features(df: pd.DataFrame) -> pd.DataFrame:
    """Re-implementation of the original group's feature set, for comparison."""
    y = df[TARGET]
    idx = df.index
    feats = {TARGET: y}
    for col in WEATHER:
        feats[col] = df[col]
    feats["hour"] = idx.hour
    feats["day_of_week"] = idx.dayofweek
    feats["month"] = idx.month
    feats["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    for lag in (1, 2, 3, 6, 12, 23):
        feats[f"lag_{lag}"] = y.shift(lag)
    for w in (3, 6, 12, 23):
        feats[f"roll_mean_{w}"] = y.shift(1).rolling(w).mean()
        feats[f"roll_std_{w}"] = y.shift(1).rolling(w).std()
    for col in WEATHER:
        for lag in (1, 3, 12, 23):
            feats[f"{col}_lag_{lag}"] = df[col].shift(lag)
    hol = is_holiday(pd.Series(idx.date, index=idx)).to_numpy()
    feats["is_holiday"] = hol.astype(int)
    return pd.DataFrame(feats, index=idx)
