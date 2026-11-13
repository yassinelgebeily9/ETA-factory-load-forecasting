"""Rolling-origin (expanding window) backtest."""
import numpy as np
import pandas as pd

from .config import TARGET
from .features import course_features, origin_features, targets


def build_table(df: pd.DataFrame):
    """Return origin features, course features and the (n, 24) target matrix."""
    origin = origin_features(df)
    course = course_features(df)
    Y = targets(df)
    valid = df[TARGET].notna() & Y.notna().all(axis=1)
    return origin[valid], course[valid], Y[valid].to_numpy()


def monthly_folds(index: pd.DatetimeIndex, n_folds: int):
    """Each fold tests one calendar month and trains on everything before it.

    Training origins stop 24 h before the test month starts, so no training
    target overlaps the test period.
    """
    months = pd.period_range(index.min(), index.max(), freq="M")
    # Use only complete months at the end of the data.
    last_full = months[-1] if index.max() >= months[-1].end_time.floor("h") else months[-2]
    test_months = [m for m in months if m <= last_full][-n_folds:]
    for m in test_months:
        start, end = m.start_time, m.end_time
        train = index <= start - pd.Timedelta(hours=24)
        test = (index >= start) & (index <= end)
        yield str(m), train, test


def backtest(models, df, n_folds=6):
    from .models import Ensemble

    origin, course, Y = build_table(df)
    folds = list(monthly_folds(origin.index, n_folds))
    preds = {}  # (model name, fold) -> forecast
    per_horizon, per_fold = {}, []
    for model in models:
        errs = []
        for fold, tr, te in folds:
            if isinstance(model, Ensemble):
                pred = np.mean([preds[(n, fold)] for n in model.member_names], axis=0)
            else:
                model.fit(origin[tr], course[tr], Y[tr])
                pred = model.predict(origin[te], course[te])
            preds[(model.name, fold)] = pred
            err = np.abs(pred - Y[te])
            errs.append(err)
            per_fold.append({"model": model.name, "fold": fold,
                             "n_origins": int(te.sum()), "mae": err.mean()})
        per_horizon[model.name] = np.vstack(errs).mean(axis=0)
    per_horizon = pd.DataFrame(per_horizon, index=pd.Index(range(1, 25), name="horizon"))
    return per_horizon, pd.DataFrame(per_fold)
