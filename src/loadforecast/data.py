"""Loading data and the Hesse public-holiday calendar."""
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from .config import TIME_COL


def load_series(path) -> pd.DataFrame:
    """Load a CSV and put it on a gap-free hourly index.

    Reindexing matters: the raw data has missing timestamps, and without it a
    `shift(24)` would silently point to the wrong hour after every gap.
    """
    df = pd.read_csv(path, parse_dates=[TIME_COL]).set_index(TIME_COL).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


@lru_cache(maxsize=None)
def hesse_holidays(year: int) -> frozenset:
    """Statutory public holidays in Hesse, Germany."""
    easter = _easter(year)
    fixed = [date(year, 1, 1), date(year, 5, 1), date(year, 10, 3),
             date(year, 12, 25), date(year, 12, 26)]
    moving = [easter + timedelta(days=d) for d in (-2, 1, 39, 50, 60)]
    # Good Friday, Easter Monday, Ascension, Whit Monday, Corpus Christi
    return frozenset(fixed + moving)


def is_holiday(dates: pd.Series) -> pd.Series:
    return dates.map(lambda d: d in hesse_holidays(d.year))
