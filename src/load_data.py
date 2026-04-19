"""
Load NSW GTFS static CSV files into pandas DataFrames.

This module reads the smaller GTFS tables we need for the journey planner.
We deliberately do not load shapes.txt here (it is very large).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Which files we load and the keys used in the returned dictionary.
# Order does not matter for loading; it is only for readability.
_GTFS_FILE_MAP: dict[str, str] = {
    "stops": "stops.txt",
    "routes": "routes.txt",
    "trips": "trips.txt",
    "stop_times": "stop_times.txt",
    "calendar": "calendar.txt",
    "calendar_dates": "calendar_dates.txt",
}


def load_gtfs_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load selected GTFS static tables from a folder into pandas DataFrames.

    Parameters
    ----------
    data_dir:
        Path to the folder that contains the GTFS .txt files (not the zip),
        e.g. "./full_greater_sydney_gtfs_static_0" or "data/full_greater_sydney_gtfs_static_0".

    Returns
    -------
    dict
        Keys: stops, routes, trips, stop_times, calendar, calendar_dates.
        Values: pandas DataFrames, one per file.

    Raises
    ------
    FileNotFoundError
        If the directory or any required file is missing.
    NotADirectoryError
        If ``data_dir`` is not a directory.
    """
    root = Path(data_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"GTFS data folder does not exist: {root}\n"
            "Check the path and that you extracted the GTFS feed into that folder."
        )

    if not root.is_dir():
        raise NotADirectoryError(f"GTFS data path is not a folder: {root}")

    missing: list[str] = []
    for key, filename in _GTFS_FILE_MAP.items():
        path = root / filename
        if not path.is_file():
            missing.append(f"  - {filename} (key: {key})")

    if missing:
        raise FileNotFoundError(
            "Missing required GTFS file(s) in "
            f"{root}:\n"
            + "\n".join(missing)
            + "\nDownload or copy the full feed into this folder."
        )

    # Load each CSV. GTFS static files are comma-separated text with a header row.
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in _GTFS_FILE_MAP.items():
        path = root / filename
        tables[key] = pd.read_csv(path, dtype=str, keep_default_na=False)

    return tables


if __name__ == "__main__":
    # Run this file directly to sanity-check loading: shapes.txt is NOT loaded.
    # Default: folder next to project root as described in the hackathon setup.
    default_data_dir = Path(__file__).resolve().parent.parent / "full_greater_sydney_gtfs_static_0"

    import sys

    data_directory = sys.argv[1] if len(sys.argv) > 1 else str(default_data_dir)

    print(f"Loading GTFS from: {data_directory}\n")
    data = load_gtfs_data(data_directory)

    for name, df in data.items():
        print(f"=== {name} ===")
        print(f"Shape (rows, columns): {df.shape}")
        print("First 3 rows:")
        print(df.head(3).to_string())
        print()
