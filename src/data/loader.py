from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "S-S1.CSV"


def load_io_vnbd(filepath=DATA_FILE):
    """
    Load an IO-VNBD smartphone dataset CSV.

    Returns:
        pandas.DataFrame: Loaded and cleaned dataset.
    """

    df = pd.read_csv(
    filepath,
    skipinitialspace=True,
    encoding="cp1252"
)

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # Convert timestamp column
    time_column = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"

    df["timestamp"] = pd.to_datetime(
        df[time_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


if __name__ == "__main__":
    data = load_io_vnbd()

    print("\n===== IO-VNBD DATASET =====")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    print("\n===== COLUMN NAMES =====")
    for i, column in enumerate(data.columns, start=1):
        print(f"{i:2}. {column}")

    print("\n===== FIRST 3 ROWS =====")
    print(data.head(3).to_string())

    print("\n===== MISSING VALUES =====")
    print(data.isna().sum())

    print("\n===== SAMPLING INTERVAL =====")
    intervals = data["timestamp"].diff().dt.total_seconds()
    print(intervals.describe())