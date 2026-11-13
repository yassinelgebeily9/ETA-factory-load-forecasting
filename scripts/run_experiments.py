"""Backtest all models and write metrics + charts to results/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loadforecast.config import RESULTS_DIR, TRAIN_CSV
from loadforecast.data import load_series
from loadforecast.evaluate import backtest
from loadforecast.models import all_models


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_series(TRAIN_CSV)
    per_horizon, per_fold = backtest(all_models(), df)

    summary = (
        per_fold.assign(weighted=per_fold.mae * per_fold.n_origins)
        .groupby("model", sort=False)
        .agg(weighted=("weighted", "sum"), n=("n_origins", "sum"))
    )
    summary["mae_kw"] = summary.weighted / summary.n
    base = summary.loc["Persistence", "mae_kw"]
    summary["vs_persistence_%"] = 100 * (1 - summary.mae_kw / base)
    summary = summary[["mae_kw", "vs_persistence_%"]].sort_values("mae_kw").round(3)

    summary.to_csv(RESULTS_DIR / "summary.csv")
    per_horizon.round(4).to_csv(RESULTS_DIR / "mae_per_horizon.csv")
    per_fold.pivot(index="fold", columns="model", values="mae").round(3).to_csv(
        RESULTS_DIR / "mae_per_fold.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in summary.index:
        ax.plot(per_horizon.index, per_horizon[name], marker="o", ms=3, label=name)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Rolling-origin backtest, Nov 2024 – Apr 2025 (6 monthly folds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mae_per_horizon.png", dpi=150)

    print(summary.to_string())
    print()
    print(per_fold.pivot(index="fold", columns="model", values="mae").round(2).to_string())


if __name__ == "__main__":
    main()
