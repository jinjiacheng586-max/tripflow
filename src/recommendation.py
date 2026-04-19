"""
Combine direct and one-transfer journey options and rank them for the user.

Direct trips use a simple time-based score; transfers reuse ``add_reliability_score``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.load_data import load_gtfs_data
    from src.preprocess import preprocess_gtfs_tables
    from src.route_search import find_direct_routes
    from src.transfer_search import find_one_transfer_routes
    from src.scoring import add_reliability_score
except ImportError:  # Running as `python src/recommendation.py`
    from load_data import load_gtfs_data
    from preprocess import preprocess_gtfs_tables
    from route_search import find_direct_routes
    from transfer_search import find_one_transfer_routes
    from scoring import add_reliability_score


def add_journey_timing_columns(
    routes: pd.DataFrame,
    earliest_departure_min: int,
) -> pd.DataFrame:
    """
    Add commuter-focused timing columns based on the user's selected start time.
    """
    out = routes.copy()
    out["wait_minutes"] = out["origin_departure_minutes"] - earliest_departure_min
    out["journey_total_minutes"] = (
        out["destination_arrival_minutes"] - earliest_departure_min
    )
    return out


def deduplicate_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate route rows using the main identity columns.

    We keep the first row because the table is already sorted in recommendation
    order before this function is called.
    """
    preferred_subset = [
        "route_type",
        "origin_stop_name",
        "destination_stop_name",
        "transfer_stop_name",
        "origin_departure_time",
        "destination_arrival_time",
        "origin_stop_id",
        "destination_stop_id",
        "transfer_stop_id",
    ]
    subset = [col for col in preferred_subset if col in routes.columns]

    if not subset:
        return routes

    return routes.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def recommend_routes(
    gtfs_data: dict,
    origin_stop_id: str,
    destination_stop_id: str,
    earliest_departure_min: int,
) -> dict:
    """
    Build a ranked list of direct and one-transfer options between two stops.

    **Direct routes** get ``buffer_label = "Direct"``, no transfer buffer, and a
    simple score ``max(0, round(10 - total_travel_minutes / 20, 1))``.

    **One-transfer routes** are scored with :func:`add_reliability_score`.

    Each route also gets:
    - ``wait_minutes`` = ``origin_departure_minutes - earliest_departure_min``
    - ``journey_total_minutes`` = ``destination_arrival_minutes - earliest_departure_min``

    Returns
    -------
    dict
        - ``all_routes``: combined table, sorted by ``journey_total_minutes``
          (shortest first), then ``reliability_score`` (highest first), then
          ``total_travel_minutes`` (shortest first), then deduplicated.
        - ``best_route``: top row using that same ranking, or ``None`` if nothing matches.
    """
    origin_stop_id = str(origin_stop_id)
    destination_stop_id = str(destination_stop_id)

    direct = find_direct_routes(
        gtfs_data,
        origin_stop_id,
        destination_stop_id,
        earliest_departure_min,
    )

    transfer = find_one_transfer_routes(
        gtfs_data,
        origin_stop_id,
        destination_stop_id,
        earliest_departure_min,
    )
    transfer = add_reliability_score(transfer)

    frames: list[pd.DataFrame] = []

    # --- Direct: align column names with transfer rows where possible ---
    if len(direct) > 0:
        d = direct.copy()
        d["total_travel_minutes"] = d["travel_minutes"]
        d["buffer_label"] = "Direct"
        d["transfer_buffer_minutes"] = pd.NA
        d["transfer_stop_id"] = pd.NA
        d["reliability_score"] = d["total_travel_minutes"].apply(
            lambda m: max(0.0, round(10.0 - float(m) / 20.0, 1))
        )
        d["route_type"] = "Direct"
        frames.append(d)

    if len(transfer) > 0:
        t = transfer.copy()
        t["route_type"] = "One Transfer"
        frames.append(t)

    if not frames:
        return {"all_routes": pd.DataFrame(), "best_route": None}

    all_routes = pd.concat(frames, ignore_index=True, sort=False)
    all_routes = add_journey_timing_columns(all_routes, earliest_departure_min)
    all_routes = all_routes.sort_values(
        by=["journey_total_minutes", "reliability_score", "total_travel_minutes"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    all_routes = deduplicate_routes(all_routes)

    best_route: pd.Series | None = (
        None if all_routes.empty else all_routes.iloc[0]
    )
    return {"all_routes": all_routes, "best_route": best_route}


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "full_greater_sydney_gtfs_static_0"

    print(f"Loading GTFS from: {data_dir}\n")
    gtfs = load_gtfs_data(str(data_dir))
    gtfs = preprocess_gtfs_tables(gtfs)

    # Sample O/D: two stops on the same trip (usually yields strong direct options)
    st = gtfs["stop_times"]
    demo_trip_id = st["trip_id"].iloc[0]
    trip_rows = st[st["trip_id"] == demo_trip_id].sort_values("stop_sequence")
    origin_stop_id = str(trip_rows.iloc[0]["stop_id"])
    destination_stop_id = str(trip_rows.iloc[min(8, len(trip_rows) - 1)]["stop_id"])

    print(
        "Demo recommendation:\n"
        f"  origin_stop_id = {origin_stop_id}\n"
        f"  destination_stop_id = {destination_stop_id}\n"
        f"  earliest_departure_min = 0\n\n"
    )

    out = recommend_routes(gtfs, origin_stop_id, destination_stop_id, 0)

    print("Best route (top row):")
    if out["best_route"] is None:
        print("  (none)\n")
    else:
        print(out["best_route"].to_string())
        print()

    display_cols = [
        "route_type",
        "origin_stop_id",
        "destination_stop_id",
        "transfer_stop_id",
        "origin_departure_time",
        "destination_arrival_time",
        "wait_minutes",
        "journey_total_minutes",
        "transfer_buffer_minutes",
        "buffer_label",
        "total_travel_minutes",
        "reliability_score",
    ]
    print("First 10 rows (all_routes):")
    # Some columns may be missing if both frames were empty; guard with reindex
    table = out["all_routes"]
    available = [c for c in display_cols if c in table.columns]
    print(table[available].head(10).to_string())
