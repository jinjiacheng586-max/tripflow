"""
Preprocess GTFS tables for the journey planner.

Turns GTFS clock times into minutes since midnight (including hours past 24),
and prepares stop_times for trip-based lookups.
"""

from __future__ import annotations

import math
from copy import copy
from pathlib import Path
from typing import Union

import pandas as pd

try:
    from src.load_data import load_gtfs_data
except ImportError:  # Running as `python src/preprocess.py` (src on sys.path)
    from load_data import load_gtfs_data


def time_to_minutes(time_str: str) -> Union[int, float]:
    """
    Convert a GTFS time string (HH:MM:SS) to minutes since midnight.

    GTFS allows hours above 24 for trips that run past midnight; we use the
    hour value as-is (e.g. 25:10:00 -> 25*60 + 10 = 1510 minutes).

    Parameters
    ----------
    time_str:
        A string like "08:15:00" or "25:10:00". Empty or invalid strings
        are treated as missing (returns NaN).

    Returns
    -------
    int or float
        Whole minutes since midnight (seconds rounded to the nearest minute),
        or ``float('nan')`` if the input is empty or cannot be parsed.
    """
    if time_str is None:
        return math.nan

    text = str(time_str).strip()
    if text == "":
        return math.nan

    # Expected format: hours:minutes:seconds (GTFS uses 24+ hour times sometimes)
    parts = text.split(":")
    if len(parts) < 2:
        return math.nan

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return math.nan

    # Whole minutes; seconds rounded so "08:15:30" becomes 496, etc.
    return int(hours * 60 + minutes + round(seconds / 60.0))


def preprocess_gtfs_tables(gtfs_data: dict) -> dict:
    """
    Build a new GTFS dict with an enriched, sorted copy of ``stop_times``.

    - Keeps every original column on ``stop_times``.
    - Adds ``arrival_minutes`` and ``departure_minutes``.
    - Converts ``stop_sequence`` to numeric values.
    - Sorts rows by ``trip_id`` then ``stop_sequence``.

    Other tables (stops, routes, trips, calendar, calendar_dates) are copied
    by reference into the returned dict unchanged.
    """
    # Shallow copy of the dict so we do not mutate the caller's mapping
    out = copy(gtfs_data)

    # Work on a true DataFrame copy so the original stop_times is untouched
    st = gtfs_data["stop_times"].copy()

    # New columns: minutes since midnight for clock times
    st["arrival_minutes"] = st["arrival_time"].map(time_to_minutes)
    st["departure_minutes"] = st["departure_time"].map(time_to_minutes)

    # stop_sequence must sort numerically (10 after 9, not after 1)
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")

    st = st.sort_values(by=["trip_id", "stop_sequence"], kind="mergesort")
    st = st.reset_index(drop=True)

    out["stop_times"] = st
    return out


if __name__ == "__main__":
    # Default GTFS folder relative to the project root (same as hackathon layout)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "full_greater_sydney_gtfs_static_0"

    print(f"Loading GTFS from: {data_dir}\n")
    gtfs = load_gtfs_data(str(data_dir))
    gtfs = preprocess_gtfs_tables(gtfs)

    cols = [
        "trip_id",
        "stop_id",
        "arrival_time",
        "departure_time",
        "arrival_minutes",
        "departure_minutes",
        "stop_sequence",
    ]
    print(gtfs["stop_times"][cols].head(10).to_string())