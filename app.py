"""
Minimal Streamlit demo for the Reliability-First Journey Planner.

Run from the project folder:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import time, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from src.load_data import load_gtfs_data
    from src.preprocess import preprocess_gtfs_tables
    from src.recommendation import recommend_routes
except ImportError:  # If the app is started with a different PYTHONPATH
    from load_data import load_gtfs_data
    from preprocess import preprocess_gtfs_tables
    from recommendation import recommend_routes

# GTFS folder next to this file: ./data/full_greater_sydney_gtfs_static_0
DATA_DIR = Path(__file__).resolve().parent / "data" / "full_greater_sydney_gtfs_static_0"

# How many stations to show in each dropdown after filtering (keeps the UI fast)
MAX_STOP_OPTIONS = 800

# In fast demo mode, search only the first few grouped platforms per station.
FAST_DEMO_MAX_PLATFORM_IDS = 12

# Friendly default for the HH:MM picker.
DEFAULT_DEPARTURE_TIME = time(8, 0)

# Preferred column order for the “all routes” preview (names first, then times and scores)
ALL_ROUTES_PRIMARY_COLS = [
    "route_type",
    "origin_stop_name",
    "destination_stop_name",
    "transfer_stop_name",
    "origin_departure_time",
    "destination_arrival_time",
    "transfer_buffer_minutes",
    "buffer_label",
    "total_travel_minutes",
    "reliability_score",
]

# Shown after the primary columns when present (helps cross-check against names)
ALL_ROUTES_ID_COLS = [
    "origin_stop_id",
    "destination_stop_id",
    "transfer_stop_id",
]


@st.cache_resource(show_spinner="Loading GTFS…")
def _load_demo_state(data_dir: str) -> tuple[dict, pd.DataFrame, dict[str, str]]:
    """Load GTFS and derive app-side metadata once per session."""
    raw = load_gtfs_data(data_dir)
    gtfs = preprocess_gtfs_tables(raw)
    stops_df = gtfs["stops"]
    station_df = build_station_groups(stops_df)
    stop_lookup = build_stop_id_to_name(stops_df)
    return gtfs, station_df, stop_lookup


def time_to_minutes_since_midnight(value: time) -> int:
    """Convert a ``datetime.time`` to whole minutes since midnight."""
    return (value.hour * 60) + value.minute


def maybe_limit_platform_ids(stop_ids: list[str], fast_demo_mode: bool) -> list[str]:
    """Optionally cap platform ids per station to keep the demo responsive."""
    if not fast_demo_mode:
        return stop_ids
    return stop_ids[:FAST_DEMO_MAX_PLATFORM_IDS]


def station_level_name(stop_name: str) -> str:
    """
    Turn a platform-level GTFS name into a station-level label.

    If the name looks like ``Something, Platform 3``, we keep only ``Something``.
    Otherwise we use the full name (e.g. bus stops without platforms).
    """
    text = str(stop_name).strip()
    marker = ", Platform"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text


def build_station_groups(stops: pd.DataFrame) -> pd.DataFrame:
    """
    One row per station-level name, with all platform ``stop_id``s grouped together.

    Columns: ``station_key`` (searchable label), ``stop_ids`` (list of strings).
    """
    s = stops[["stop_id", "stop_name"]].copy()
    s["stop_id"] = s["stop_id"].astype(str)
    s["station_key"] = s["stop_name"].map(station_level_name)
    grouped = (
        s.groupby("station_key", as_index=False)
        .agg(stop_ids=("stop_id", lambda ids: sorted(set(ids.astype(str)))))
    )
    return grouped


def build_stop_id_to_name(stops: pd.DataFrame) -> dict[str, str]:
    """Map ``stop_id`` -> ``stop_name`` from the GTFS ``stops`` table."""
    return dict(zip(stops["stop_id"].astype(str), stops["stop_name"].astype(str)))


def _match_tier(stop_name: str, query_lower: str) -> int:
    """
    How strongly the station name matches the search text (higher is better).

    3 = exact whole name, 2 = name starts with the query, 1 = query appears elsewhere.
    """
    name_lower = stop_name.lower()
    if name_lower == query_lower:
        return 3
    if name_lower.startswith(query_lower):
        return 2
    if query_lower in name_lower:
        return 1
    return 0


def _looks_like_station(stop_name: str) -> int:
    """1 if the name looks like a train station (contains the word Station)."""
    return 1 if "station" in stop_name.lower() else 0


def build_stop_choices_for_query(
    stations_df: pd.DataFrame,
    query: str,
    max_results: int = MAX_STOP_OPTIONS,
) -> tuple[list[str], dict[str, list[str] | None]]:
    """
    Build selectbox options (station-level names) and a station -> platform ids map.

    - Filters by substring on ``station_key`` when ``query`` is non-empty.
    - When searching, ranks matches so **stations** and **better text matches**
      (exact / prefix before loose substring) appear first.
    - With an empty query, shows the first ``max_results`` stations sorted by name (A→Z).
    """
    s = stations_df.copy()

    q = query.strip()
    if q:
        ql = q.lower()
        mask = s["station_key"].str.lower().str.contains(ql, na=False, regex=False)
        s = s[mask]
        s["_tier"] = s["station_key"].map(lambda n: _match_tier(n, ql))
        s["_station"] = s["station_key"].map(_looks_like_station)
        s = s.sort_values(
            by=["_tier", "_station", "station_key"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(max_results)
        s = s.drop(columns=["_tier", "_station"])
    else:
        s = s.sort_values("station_key", kind="mergesort").head(max_results)

    if s.empty:
        placeholder = "— No matching stations —"
        return [placeholder], {placeholder: None}

    labels = s["station_key"].tolist()
    mapping: dict[str, list[str] | None] = {
        row["station_key"]: list(row["stop_ids"]) for _, row in s.iterrows()
    }
    return labels, mapping


def search_routes_all_platform_pairs(
    gtfs: dict,
    origin_stop_ids: list[str],
    destination_stop_ids: list[str],
    earliest_departure_min: int,
) -> dict:
    """
    Call :func:`recommend_routes` for every origin platform × destination platform pair,
    merge all non-empty results, re-sort, and return the same dict shape as
    ``recommend_routes`` (``all_routes`` + ``best_route``).
    """
    frames: list[pd.DataFrame] = []

    for oid in origin_stop_ids:
        for did in destination_stop_ids:
            chunk = recommend_routes(gtfs, oid, did, earliest_departure_min)["all_routes"]
            if chunk is not None and len(chunk) > 0:
                frames.append(chunk)

    if not frames:
        return {"all_routes": pd.DataFrame(), "best_route": None}

    all_routes = pd.concat(frames, ignore_index=True, sort=False)
    all_routes = all_routes.drop_duplicates()

    all_routes = all_routes.sort_values(
        by=["reliability_score", "total_travel_minutes"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    best_route = None if all_routes.empty else all_routes.iloc[0]
    return {"all_routes": all_routes, "best_route": best_route}


def stop_name_for(lookup: dict[str, str], stop_id) -> str:
    """Return a display name, or empty text when the id is missing / unknown."""
    if stop_id is None:
        return ""
    try:
        if pd.isna(stop_id):
            return ""
    except (TypeError, ValueError):
        pass
    key = str(stop_id).strip()
    if key == "" or key.lower() == "nan":
        return ""
    return str(lookup.get(key, ""))


def add_stop_name_columns(df: pd.DataFrame, lookup: dict[str, str]) -> pd.DataFrame:
    """Add ``*_stop_name`` columns next to the ``*_stop_id`` columns when present."""
    out = df.copy()
    pairs = [
        ("origin_stop_id", "origin_stop_name"),
        ("destination_stop_id", "destination_stop_name"),
        ("transfer_stop_id", "transfer_stop_name"),
    ]
    for id_col, name_col in pairs:
        if id_col in out.columns:
            out[name_col] = out[id_col].map(lambda x: stop_name_for(lookup, x))
        else:
            out[name_col] = ""
    return out


def ordered_route_display_columns(df: pd.DataFrame) -> list[str]:
    """Build a stable column list: primary order first, then optional stop ids."""
    cols: list[str] = []
    for c in ALL_ROUTES_PRIMARY_COLS:
        if c in df.columns and c not in cols:
            cols.append(c)
    for c in ALL_ROUTES_ID_COLS:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def _has_nonempty_value(row: pd.Series, key: str) -> bool:
    """True if the column exists and has a real value (not NA / blank)."""
    if key not in row.index:
        return False
    v = row[key]
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return str(v).strip() != ""


def _format_transfer_buffer_minutes(row: pd.Series) -> str:
    """Show minutes, or N/A when the value is missing."""
    if "transfer_buffer_minutes" not in row.index:
        return "N/A"
    v = row["transfer_buffer_minutes"]
    try:
        if pd.isna(v):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    return str(v)


def buffer_status_colors(label: str) -> tuple[str, str]:
    """Return background and text colors for a buffer status label."""
    palette = {
        "Safe": ("#dcfce7", "#166534"),
        "Risky": ("#fef3c7", "#a16207"),
        "Very Tight": ("#fee2e2", "#b91c1c"),
        "Direct": ("#dbeafe", "#1d4ed8"),
    }
    return palette.get(str(label).strip(), ("#e5e7eb", "#374151"))


def format_buffer_status_badge(label: str) -> str:
    """Build a small colored badge for the Best route card."""
    bg, fg = buffer_status_colors(label)
    return (
        f"<span style='background:{bg}; color:{fg}; padding:0.2rem 0.55rem; "
        "border-radius:999px; font-weight:600; display:inline-block;'>"
        f"{escape(str(label))}</span>"
    )


def style_buffer_label_cell(value) -> str:
    """Apply simple color styling to the buffer_label table column."""
    bg, fg = buffer_status_colors(str(value))
    return (
        f"background-color: {bg}; color: {fg}; font-weight: 600; "
        "border-radius: 999px;"
    )


def format_info_tile(label: str, value: str, background: str = "#f8fafc") -> str:
    """Render a compact info tile for the Best route card."""
    return (
        f"<div style='background:{background}; border:1px solid #e5e7eb; "
        "border-radius:14px; padding:0.8rem 0.9rem; margin-bottom:0.75rem;'>"
        f"<div style='font-size:0.8rem; color:#6b7280; margin-bottom:0.2rem;'>"
        f"{escape(label)}</div>"
        f"<div style='font-size:1rem; font-weight:600; color:#111827;'>"
        f"{escape(value)}</div>"
        "</div>"
    )


def render_best_route_card(row: pd.Series) -> None:
    """
    Display the top route in a compact, easier-to-scan recommendation card.

    Only includes lines for fields that exist; ``Transfer at`` is omitted when
    there is no transfer stop name.
    """
    try:
        card = st.container(border=True)
    except TypeError:
        card = st.container()

    with card:
        st.markdown("##### Top pick")
        top_cols = st.columns([1.2, 2.4, 1.2, 1.2])

        route_type = str(row["route_type"]) if _has_nonempty_value(row, "route_type") else "N/A"
        origin_name = (
            str(row["origin_stop_name"]) if _has_nonempty_value(row, "origin_stop_name") else "N/A"
        )
        destination_name = (
            str(row["destination_stop_name"])
            if _has_nonempty_value(row, "destination_stop_name")
            else "N/A"
        )
        departure_time = (
            str(row["origin_departure_time"])
            if _has_nonempty_value(row, "origin_departure_time")
            else "N/A"
        )
        arrival_time = (
            str(row["destination_arrival_time"])
            if _has_nonempty_value(row, "destination_arrival_time")
            else "N/A"
        )
        transfer_stop = (
            str(row["transfer_stop_name"])
            if _has_nonempty_value(row, "transfer_stop_name")
            else "No transfer"
        )
        buffer_minutes = _format_transfer_buffer_minutes(row)
        buffer_status = (
            str(row["buffer_label"]) if _has_nonempty_value(row, "buffer_label") else "N/A"
        )
        total_travel = (
            f"{row['total_travel_minutes']} minutes"
            if _has_nonempty_value(row, "total_travel_minutes")
            else "N/A"
        )
        reliability = (
            str(row["reliability_score"])
            if _has_nonempty_value(row, "reliability_score")
            else "N/A"
        )

        with top_cols[0]:
            st.markdown(
                format_info_tile("Route type", route_type, background="#eef6ff"),
                unsafe_allow_html=True,
            )
        with top_cols[1]:
            st.markdown(
                format_info_tile("Journey", f"{origin_name} -> {destination_name}"),
                unsafe_allow_html=True,
            )
        with top_cols[2]:
            st.markdown(
                format_info_tile("Departure", departure_time, background="#f9fafb"),
                unsafe_allow_html=True,
            )
        with top_cols[3]:
            st.markdown(
                format_info_tile("Arrival", arrival_time, background="#f9fafb"),
                unsafe_allow_html=True,
            )

        detail_cols = st.columns(2)

        with detail_cols[0]:
            st.markdown(
                format_info_tile("From", origin_name),
                unsafe_allow_html=True,
            )
            st.markdown(
                format_info_tile("To", destination_name),
                unsafe_allow_html=True,
            )
            st.markdown(
                format_info_tile("Transfer at", transfer_stop),
                unsafe_allow_html=True,
            )

        with detail_cols[1]:
            st.markdown(
                format_info_tile("Transfer buffer", buffer_minutes, background="#fffdf5"),
                unsafe_allow_html=True,
            )
            st.markdown(
                (
                    "<div style='background:#ffffff; border:1px solid #e5e7eb; "
                    "border-radius:14px; padding:0.8rem 0.9rem; margin-bottom:0.75rem;'>"
                    "<div style='font-size:0.8rem; color:#6b7280; margin-bottom:0.35rem;'>"
                    "Buffer status</div>"
                    f"{format_buffer_status_badge(buffer_status)}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                format_info_tile("Total travel time", total_travel, background="#f0fdf4"),
                unsafe_allow_html=True,
            )
            st.markdown(
                format_info_tile("Reliability score", reliability, background="#f5f3ff"),
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="TripFlow", layout="wide")

    st.title("TripFlow")
    st.markdown(
        "Explore **direct** and **one-transfer** options between two NSW stops. "
        "Scores favour comfortable transfer windows and shorter total travel time."
    )

    if not DATA_DIR.is_dir():
        st.error(
            f"GTFS folder not found: `{DATA_DIR}`. "
            "Place the feed under `data/full_greater_sydney_gtfs_static_0`."
        )
        return

    gtfs, station_df, stop_lookup = _load_demo_state(str(DATA_DIR))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Origin**")
        origin_search = st.text_input(
            "Search origin stations",
            placeholder="e.g. Redfern Station",
            key="origin_search",
            label_visibility="collapsed",
        )
        origin_labels, origin_station_to_ids = build_stop_choices_for_query(
            station_df, origin_search
        )
        origin_choice = st.selectbox(
            "Origin station",
            options=origin_labels,
            index=0,
            key="origin_select",
        )
        origin_stop_ids = origin_station_to_ids.get(origin_choice)

    with col2:
        st.markdown("**Destination**")
        dest_search = st.text_input(
            "Search destination stations",
            placeholder="e.g. Burwood Station",
            key="dest_search",
            label_visibility="collapsed",
        )
        dest_labels, dest_station_to_ids = build_stop_choices_for_query(
            station_df, dest_search
        )
        dest_choice = st.selectbox(
            "Destination station",
            options=dest_labels,
            index=0,
            key="dest_select",
        )
        destination_stop_ids = dest_station_to_ids.get(dest_choice)

    with col3:
        st.markdown("**Time**")
        departure_time = st.time_input(
            "Earliest departure (HH:MM)",
            value=DEFAULT_DEPARTURE_TIME,
            step=timedelta(minutes=5),
            help="Routes will start at or after this time.",
        )
        fast_demo_mode = st.checkbox(
            "Fast demo mode",
            value=True,
            help=(
                "Keeps the demo snappy by limiting how many grouped platforms "
                "per station are searched."
            ),
        )

    earliest_departure_min = time_to_minutes_since_midnight(departure_time)

    st.caption(
        f"{len(station_df):,} station groups loaded (platforms grouped). "
        f"Up to {MAX_STOP_OPTIONS} stations per dropdown — search ranks **Station** "
        f"names and closer matches higher. Searching from **{departure_time.strftime('%H:%M')}**. "
        + (
            f"Fast demo mode searches up to {FAST_DEMO_MAX_PLATFORM_IDS} platforms per station."
            if fast_demo_mode
            else "Fast demo mode is off, so all grouped platforms are searched."
        )
    )

    find_clicked = st.button("Find Routes", type="primary")

    if not find_clicked:
        return

    if (
        origin_stop_ids is None
        or destination_stop_ids is None
        or len(origin_stop_ids) == 0
        or len(destination_stop_ids) == 0
    ):
        st.warning(
            "Choose a valid origin and destination station from the lists "
            '(not the “no match” line).'
        )
        return

    search_origin_stop_ids = maybe_limit_platform_ids(origin_stop_ids, fast_demo_mode)
    search_destination_stop_ids = maybe_limit_platform_ids(
        destination_stop_ids, fast_demo_mode
    )

    with st.spinner("Searching routes across platforms…"):
        result = search_routes_all_platform_pairs(
            gtfs,
            search_origin_stop_ids,
            search_destination_stop_ids,
            earliest_departure_min,
        )

    all_routes = result["all_routes"]
    best = result["best_route"]

    if all_routes is None or len(all_routes) == 0:
        st.info(
            "No routes found for this pair and time window. "
            "Try other stations, a later earliest departure, or check the GTFS feed."
        )
        return

    st.subheader("Best route")
    if best is not None:
        best_row = add_stop_name_columns(best.to_frame().T, stop_lookup).iloc[0]
        render_best_route_card(best_row)
    else:
        st.caption("None")

    st.subheader("All routes")
    st.caption("Showing top 20 route options")

    enriched = add_stop_name_columns(all_routes, stop_lookup)
    display_cols = ordered_route_display_columns(enriched)
    if not display_cols:
        st.warning("No expected columns were found to display; showing the first 20 rows as-is.")
        preview = enriched.head(20)
    else:
        preview = enriched[display_cols].head(20)

    if "buffer_label" in preview.columns:
        preview = preview.style.map(style_buffer_label_cell, subset=["buffer_label"])

    st.dataframe(preview, use_container_width=True, hide_index=True)


main()
