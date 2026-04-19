"""
Direct route search (no transfers): trips that visit both stops in order.

Uses preprocessed ``stop_times`` with ``departure_minutes``, ``arrival_minutes``,
and numeric ``stop_sequence``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.load_data import load_gtfs_data
    from src.preprocess import preprocess_gtfs_tables
except ImportError:  # Running as `python src/route_search.py`
    from load_data import load_gtfs_data
    from preprocess import preprocess_gtfs_tables


def find_direct_routes(
    gtfs_data: dict,
    origin_stop_id: str,
    destination_stop_id: str,
    earliest_departure_min: int,
) -> pd.DataFrame:
    """
    Find non-transfer trips where you can board at the origin and alight at
    the destination, in that order along the trip.

    Parameters
    ----------
    gtfs_data:
        GTFS dict from ``load_gtfs_data``, after ``preprocess_gtfs_tables``.
    origin_stop_id, destination_stop_id:
        ``stop_id`` values (strings, same format as in ``stops.txt``).
    earliest_departure_min:
        Earliest allowed *departure* from the origin, in minutes since midnight
        (same scale as ``departure_minutes``).

    Returns
    -------
    pandas.DataFrame
        One row per matching ``trip_id``, sorted by ``origin_departure_minutes``.
        Empty if there are no direct options.
    """
    st = gtfs_data["stop_times"]

    # All stop-time rows for the origin and destination (may share many trips)
    origin_rows = st[st["stop_id"] == origin_stop_id].copy()
    dest_rows = st[st["stop_id"] == destination_stop_id].copy()

    # Keep only the columns we need so the merge does not create _x/_y duplicates
    origin_rows = origin_rows.rename(
        columns={
            "departure_time": "origin_departure_time",
            "departure_minutes": "origin_departure_minutes",
            "stop_sequence": "origin_stop_sequence",
        }
    )[
        [
            "trip_id",
            "origin_departure_time",
            "origin_departure_minutes",
            "origin_stop_sequence",
        ]
    ]

    dest_rows = dest_rows.rename(
        columns={
            "arrival_time": "destination_arrival_time",
            "arrival_minutes": "destination_arrival_minutes",
            "stop_sequence": "destination_stop_sequence",
        }
    )[
        [
            "trip_id",
            "destination_arrival_time",
            "destination_arrival_minutes",
            "destination_stop_sequence",
        ]
    ]

    # One row per trip where both stops exist (multiple rows if duplicate trip_ids
    # with different stop_sequence pairs — rare; we dedupe later)
    merged = origin_rows.merge(dest_rows, on="trip_id", how="inner")

    # Origin must appear strictly before destination along the trip
    merged = merged[merged["origin_stop_sequence"] < merged["destination_stop_sequence"]]

    # Respect earliest departure at origin (ignore rows with missing times)
    merged = merged[pd.notna(merged["origin_departure_minutes"])]
    merged = merged[pd.notna(merged["destination_arrival_minutes"])]
    merged = merged[merged["origin_departure_minutes"] >= earliest_departure_min]

    # One row per trip_id: if duplicates exist, keep the earliest departure
    merged = merged.sort_values("origin_departure_minutes", kind="mergesort")
    merged = merged.drop_duplicates(subset=["trip_id"], keep="first")

    merged["travel_minutes"] = (
        merged["destination_arrival_minutes"] - merged["origin_departure_minutes"]
    )

    # Final public columns (stop ids repeated for clarity in the result table)
    out = pd.DataFrame(
        {
            "trip_id": merged["trip_id"],
            "origin_stop_id": origin_stop_id,
            "destination_stop_id": destination_stop_id,
            "origin_departure_time": merged["origin_departure_time"],
            "destination_arrival_time": merged["destination_arrival_time"],
            "origin_departure_minutes": merged["origin_departure_minutes"],
            "destination_arrival_minutes": merged["destination_arrival_minutes"],
            "travel_minutes": merged["travel_minutes"],
        }
    )

    out = out.sort_values("origin_departure_minutes", kind="mergesort").reset_index(drop=True)
    return out


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "full_greater_sydney_gtfs_static_0"

    print(f"Loading GTFS from: {data_dir}\n")
    gtfs = load_gtfs_data(str(data_dir))
    gtfs = preprocess_gtfs_tables(gtfs)

    stops = gtfs["stops"]
    # Pick two stop_ids that definitely appear on the same trip so the demo
    # usually returns results: take any trip, then first and a later stop on it.
    st = gtfs["stop_times"]
    demo_trip_id = st["trip_id"].iloc[0]
    trip_rows = st[st["trip_id"] == demo_trip_id].sort_values("stop_sequence")
    origin_stop_id = str(trip_rows.iloc[0]["stop_id"])
    destination_stop_id = str(trip_rows.iloc[min(8, len(trip_rows) - 1)]["stop_id"])

    # Show that both IDs exist in the published stops table
    assert origin_stop_id in set(stops["stop_id"].astype(str))
    assert destination_stop_id in set(stops["stop_id"].astype(str))

    print(
        "Demo search (sample stop_ids taken from a real trip, "
        f"both present in stops.txt):\n"
        f"  origin_stop_id = {origin_stop_id}\n"
        f"  destination_stop_id = {destination_stop_id}\n"
        f"  earliest_departure_min = 0\n\n"
        "First 10 direct routes:\n"
    )

    results = find_direct_routes(gtfs, origin_stop_id, destination_stop_id, 0)
    cols = [
        "trip_id",
        "origin_stop_id",
        "destination_stop_id",
        "origin_departure_time",
        "destination_arrival_time",
        "origin_departure_minutes",
        "destination_arrival_minutes",
        "travel_minutes",
    ]
    print(results[cols].head(10).to_string())
