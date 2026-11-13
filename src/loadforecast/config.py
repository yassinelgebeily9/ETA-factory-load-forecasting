"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_SLICES_DIR = DATA_DIR / "test_slices"

TIME_COL = "Time"
TARGET = "Elec_P_LVMDB_ETA_kW"
WEATHER = [
    "humidity_percent",
    "precipitation_height_mm",
    "temperature_air_mean_C",
    "wind_speed_m_per_s",
]

# At prediction time the model only sees the last 24 hourly rows
# (this mirrors the test slices of the challenge).
WINDOW = 24
HORIZONS = list(range(1, 25))

SEED = 42
