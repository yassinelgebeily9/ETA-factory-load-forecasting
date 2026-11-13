"""Features at origin t must not change when data after t changes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loadforecast.config import TARGET, WEATHER
from loadforecast.features import course_features, horizon_features, origin_features


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    cols = {TARGET: rng.uniform(2, 20, n), **{c: rng.normal(size=n) for c in WEATHER}}
    return pd.DataFrame(cols, index=idx)


def test_features_ignore_the_future():
    df = _toy()
    t = df.index[120]
    changed = df.copy()
    changed.loc[changed.index > t] *= 100  # scramble everything after t

    for fn in (origin_features, course_features):
        a, b = fn(df).loc[t], fn(changed).loc[t]
        pd.testing.assert_series_equal(a, b)

    o_a, o_b = origin_features(df), origin_features(changed)
    for h in (1, 12, 24):
        pd.testing.assert_frame_equal(
            horizon_features(o_a.index, o_a, h).loc[[t]],
            horizon_features(o_b.index, o_b, h).loc[[t]],
        )
