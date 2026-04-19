"""
One-transfer route search: board at origin, alight at a transfer stop, then
continue on a second trip to the destination.

Uses preprocessed ``stop_times`` (minutes columns and numeric ``stop_sequence``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.load_data import load_gtfs_data
    from src.preprocess import preprocess_gtfs_tables
except ImportError:  # Running as `python src/transfer_search.py`
    from load_data import load_gtfs_data
    from preprocess import preprocess_gtfs_tables


def classify_transfer_buffer(buffer_minutes: int) -> str:
    """
    Turn a transfer wait time (minutes) into a short reliability-style label.

    Rules:
    - 10 minutes or more  -> "Safe"
    - 5 to 9 minutes      -> "Risky"
    - under 5 minutes     -> "Very Tight"
    """
    if buffer_minutes >= 10:
        return "Safe"
    if buffer_minutes >= 5:
        return "Risky"
    return "Very Tight"


def find_one_transfer_routes(
    gtfs_data: dict,
    origin_stop_id: str,
    destination_stop_id: str,
    earliest_departure_min: int,
    min_transfer_buffer: int = 3,
    max_transfer_buffer: int = 30,
) -> pd.DataFrame:
    """
    Find journeys with exactly one transfer between two distinct trips.

    First leg: board at ``origin_stop_id`` and alight at ``transfer_stop_id``.
    Second leg: board at the same ``transfer_stop_id`` and alight at
    ``destination_stop_id``. The two trips must be different.

    The *transfer buffer* is the waiting time at the transfer stop:
    ``second_leg_departure_minutes - first_leg_arrival_minutes``. It must lie
    between ``min_transfer_buffer`` and ``max_transfer_buffer`` (inclusive).

    Parameters
    ----------
    gtfs_data:
        GTFS dict from ``load_gtfs_data``, after ``preprocess_gtfs_tables``.
    origin_stop_id, destination_stop_id:
        ``stop_id`` values as strings.
    earliest_departure_min:
        Earliest allowed departure from the origin (minutes since midnight).
    min_transfer_buffer, max_transfer_buffer:
        Allowed waiting time at the transfer (minutes), inclusive on both ends.

    Returns
    -------
    pandas.DataFrame
        One row per valid one-transfer itinerary, sorted by ``total_travel_minutes``.
        Includes ``buffer_label`` from :func:`classify_transfer_buffer`.
    """
    origin_stop_id = str(origin_stop_id)
    destination_stop_id = str(destination_stop_id)

    st = gtfs_data["stop_times"]

    # --- First leg: origin -> transfer (same trip, transfer after origin) ---
    o = st[st["stop_id"] == origin_stop_id][
        ["trip_id", "stop_sequence", "departure_time", "departure_minutes"]
    ].rename(
        columns={
            "stop_sequence": "o_seq",
            "departure_time": "origin_departure_time",
            "departure_minutes": "origin_departure_minutes",
        }
    )

    x = st[
        ["trip_id", "stop_sequence", "stop_id", "arrival_time", "arrival_minutes"]
    ].rename(
        columns={
            "stop_sequence": "t_seq",
            "stop_id": "transfer_stop_id",
            "arrival_time": "first_leg_arrival_time",
            "arrival_minutes": "first_leg_arrival_minutes",
        }
    )

    leg1 = o.merge(x, on="trip_id", how="inner")
    leg1 = leg1[leg1["t_seq"] > leg1["o_seq"]]
    leg1 = leg1[leg1["transfer_stop_id"] != origin_stop_id]
    leg1 = leg1[leg1["origin_departure_minutes"] >= earliest_departure_min]
    leg1 = leg1[pd.notna(leg1["first_leg_arrival_minutes"])]
    leg1 = leg1.rename(columns={"trip_id": "first_trip_id"})

    # --- Second leg: transfer -> destination (same trip, destination after transfer) ---
    tboard = st[
        ["trip_id", "stop_sequence", "stop_id", "departure_time", "departure_minutes"]
    ].rename(
        columns={
            "stop_sequence": "ts_seq",
            "stop_id": "transfer_stop_id",
            "departure_time": "second_leg_departure_time",
            "departure_minutes": "second_leg_departure_minutes",
        }
    )

    d = st[st["stop_id"] == destination_stop_id][
        ["trip_id", "stop_sequence", "arrival_time", "arrival_minutes"]
    ].rename(
        columns={
            "stop_sequence": "d_seq",
            "arrival_time": "destination_arrival_time",
            "arrival_minutes": "destination_arrival_minutes",
        }
    )

    leg2 = tboard.merge(d, on="trip_id", how="inner")
    leg2 = leg2[leg2["ts_seq"] < leg2["d_seq"]]
    leg2 = leg2[leg2["transfer_stop_id"] != destination_stop_id]
    leg2 = leg2[pd.notna(leg2["second_leg_departure_minutes"])]
    leg2 = leg2[pd.notna(leg2["destination_arrival_minutes"])]
    leg2 = leg2.rename(columns={"trip_id": "second_trip_id"})

    # --- Join legs at the transfer stop (same physical ``stop_id``) ---
    combo = leg1.merge(leg2, on="transfer_stop_id", how="inner")

    # Degenerate transfer point: must differ from origin and destination
    combo = combo[combo["transfer_stop_id"] != origin_stop_id]
    combo = combo[combo["transfer_stop_id"] != destination_stop_id]

    # Two distinct vehicle trips (staying on one vehicle is a direct trip, not a transfer)
    combo = combo[combo["first_trip_id"] != combo["second_trip_id"]]

    combo["transfer_buffer_minutes"] = (
        combo["second_leg_departure_minutes"] - combo["first_leg_arrival_minutes"]
    )

    combo = combo[
        (combo["transfer_buffer_minutes"] >= min_transfer_buffer)
        & (combo["transfer_buffer_minutes"] <= max_transfer_buffer)
    ]

    combo["total_travel_minutes"] = (
        combo["destination_arrival_minutes"] - combo["origin_departure_minutes"]
    )
    combo = combo[combo["total_travel_minutes"] >= 0]

    out = pd.DataFrame(
        {
            "first_trip_id": combo["first_trip_id"],
            "second_trip_id": combo["second_trip_id"],
            "origin_stop_id": origin_stop_id,
            "transfer_stop_id": combo["transfer_stop_id"],
            "destination_stop_id": destination_stop_id,
            "origin_departure_time": combo["origin_departure_time"],
            "first_leg_arrival_time": combo["first_leg_arrival_time"],
            "second_leg_departure_time": combo["second_leg_departure_time"],
            "destination_arrival_time": combo["destination_arrival_time"],
            "origin_departure_minutes": combo["origin_departure_minutes"],
            "first_leg_arrival_minutes": combo["first_leg_arrival_minutes"],
            "second_leg_departure_minutes": combo["second_leg_departure_minutes"],
            "destination_arrival_minutes": combo["destination_arrival_minutes"],
            "transfer_buffer_minutes": combo["transfer_buffer_minutes"],
            "total_travel_minutes": combo["total_travel_minutes"],
        }
    )

    out = out.sort_values("total_travel_minutes", kind="mergesort")
    out = out.drop_duplicates(
        subset=[
            "first_trip_id",
            "second_trip_id",
            "transfer_stop_id",
            "origin_departure_minutes",
            "second_leg_departure_minutes",
        ],
        keep="first",
    )
    out = out.reset_index(drop=True)

    # Human-readable transfer comfort (rounded whole minutes match the labels)
    out["buffer_label"] = out["transfer_buffer_minutes"].apply(
        lambda m: classify_transfer_buffer(int(round(float(m))))
    )

    return out


def _pick_demo_od_with_connection(
    gtfs_data: dict,
    min_transfer_buffer: int,
    max_transfer_buffer: int,
) -> tuple[str, str] | None:
    """
    Try to find an origin/destination pair that yields at least one one-transfer
    result with the given buffer window (used only for the demo block).
    """
    st = gtfs_data["stop_times"]
    # Prefer busy interchange stops so two different trips are likely to meet
    # in a realistic time window.
    for transfer_stop_id in st["stop_id"].value_counts().head(80).index.astype(str):
        at = st[st["stop_id"] == transfer_stop_id]
        if len(at) < 2:
            continue
        # Compare pairs of visits at this stop on different trips (cap work)
        sample = at.sort_values("departure_minutes").head(400)
        for _, r_in in sample.iterrows():
            for _, r_out in sample.iterrows():
                if r_in["trip_id"] == r_out["trip_id"]:
                    continue
                if pd.isna(r_in["arrival_minutes"]) or pd.isna(r_out["departure_minutes"]):
                    continue
                buf = float(r_out["departure_minutes"]) - float(r_in["arrival_minutes"])
                if not (min_transfer_buffer <= buf <= max_transfer_buffer):
                    continue

                t1 = r_in["trip_id"]
                t2 = r_out["trip_id"]
                seg1 = st[st["trip_id"] == t1].sort_values("stop_sequence")
                row_t1 = seg1[seg1["stop_id"] == transfer_stop_id].iloc[0]
                earlier = seg1[seg1["stop_sequence"] < row_t1["stop_sequence"]]
                if earlier.empty:
                    continue
                origin = str(earlier.iloc[0]["stop_id"])

                seg2 = st[st["trip_id"] == t2].sort_values("stop_sequence")
                row_t2 = seg2[seg2["stop_id"] == transfer_stop_id].iloc[0]
                later = seg2[seg2["stop_sequence"] > row_t2["stop_sequence"]]
                if later.empty:
                    continue
                dest = str(later.iloc[0]["stop_id"])

                if origin == dest:
                    continue

                res = find_one_transfer_routes(
                    gtfs_data,
                    origin,
                    dest,
                    0,
                    min_transfer_buffer,
                    max_transfer_buffer,
                )
                if len(res) > 0:
                    return origin, dest
    return None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "full_greater_sydney_gtfs_static_0"

    print(f"Loading GTFS from: {data_dir}\n")
    gtfs = load_gtfs_data(str(data_dir))
    gtfs = preprocess_gtfs_tables(gtfs)

    min_b, max_b = 3, 30
    picked = _pick_demo_od_with_connection(gtfs, min_b, max_b)

    if picked is None:
        print(
            "No quick auto-pair with a 3–30 minute transfer window; "
            "running a manual example with a relaxed 3–90 minute window.\n"
        )
        min_b, max_b = 3, 90
        picked = _pick_demo_od_with_connection(gtfs, min_b, max_b)

    if picked is None:
        # Last resort: still run the API with fixed IDs (may return 0 rows).
        origin_stop_id = str(gtfs["stops"]["stop_id"].iloc[0])
        destination_stop_id = str(gtfs["stops"]["stop_id"].iloc[1])
        print(
            "Using first two stops from stops.txt as a fallback demo "
            f"(origin={origin_stop_id}, dest={destination_stop_id}).\n"
        )
    else:
        origin_stop_id, destination_stop_id = picked
        print(
            "Demo one-transfer search (auto-picked O/D that connect within "
            f"{min_b}–{max_b} minutes at some transfer stop):\n"
            f"  origin_stop_id = {origin_stop_id}\n"
            f"  destination_stop_id = {destination_stop_id}\n"
            f"  earliest_departure_min = 0\n"
            f"  min_transfer_buffer = {min_b}\n"
            f"  max_transfer_buffer = {max_b}\n\n"
            "First 10 one-transfer routes:\n"
        )

    results = find_one_transfer_routes(
        gtfs,
        origin_stop_id,
        destination_stop_id,
        0,
        min_b,
        max_b,
    )
    display_cols = [
        "origin_stop_id",
        "transfer_stop_id",
        "destination_stop_id",
        "origin_departure_time",
        "first_leg_arrival_time",
        "second_leg_departure_time",
        "destination_arrival_time",
        "transfer_buffer_minutes",
        "buffer_label",
        "total_travel_minutes",
    ]
    print(results[display_cols].head(10).to_string())
