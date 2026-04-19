"""
Reliability-style scores for journey options (e.g. one-transfer routes).

Scores are simple heuristics: comfortable transfers and shorter trips score higher.
"""

from __future__ import annotations

import math

import pandas as pd


def calculate_reliability_score(buffer_label: str, total_travel_minutes: float) -> float:
    """
    Map a transfer comfort label and total trip time to a score from 0–10.

    - Start at **10.0**.
    - Penalise tighter transfer windows (``buffer_label``).
    - Penalise longer total travel time: **0.5** points per **10 minutes** of
      ``total_travel_minutes`` (counting whole 10-minute blocks downward).
    - Clamp the result so it is not below **0**.
    - Return the score rounded to **one** decimal place.

    Missing or invalid ``total_travel_minutes`` is treated as **no extra time
    penalty** (0 whole 10-minute blocks), so the score still reflects the
    buffer label only.

    Parameters
    ----------
    buffer_label:
        One of ``"Safe"``, ``"Risky"``, or ``"Very Tight"`` (see
        ``classify_transfer_buffer`` in ``transfer_search``). Empty string if unknown.
    total_travel_minutes:
        Door-to-door time in minutes (may be fractional).
    """
    score = 10.0

    if buffer_label == "Safe":
        score -= 0
    elif buffer_label == "Risky":
        score -= 2
    elif buffer_label == "Very Tight":
        score -= 4
    # Any other label: no extra penalty (keeps the demo forgiving)

    try:
        tt = float(total_travel_minutes)
    except (TypeError, ValueError):
        tt = float("nan")

    # math.floor does not accept NaN; treat missing/invalid time as no duration penalty
    if math.isnan(tt) or math.isinf(tt):
        ten_minute_blocks = 0
    else:
        ten_minute_blocks = math.floor(tt / 10.0)

    score -= 0.5 * ten_minute_blocks

    score = max(0.0, score)
    return round(score, 1)


def _normalize_buffer_label(value) -> str:
    """Turn cell values into a plain string label; missing -> empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _scalar_cell(row: pd.Series, col: str):
    """
    One value from a row, even if the table has duplicate column names
    (``row[col]`` can be a small Series).
    """
    if col not in row.index:
        return float("nan")
    val = row[col]
    if isinstance(val, pd.Series):
        return val.iloc[0]
    return val


def add_reliability_score(routes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a single numeric ``reliability_score`` column to a routes table.

    Expects ``buffer_label`` and ``total_travel_minutes`` when those fields exist.
    Returns a **new** DataFrame so the original is left unchanged.

    Uses a simple row loop so the result is always one scalar score per row
    (avoids issues with duplicate column names or ``apply`` returning a DataFrame).
    """
    out = routes_df.copy()

    # Empty input: keep shape, add an empty score column so callers can rely on it
    if out.empty:
        out["reliability_score"] = pd.Series(dtype="float64")
        return out

    scores: list[float] = []
    for idx in range(len(out)):
        row = out.iloc[idx]
        label = _normalize_buffer_label(_scalar_cell(row, "buffer_label"))
        travel_raw = _scalar_cell(row, "total_travel_minutes")
        try:
            travel = float(travel_raw)
        except (TypeError, ValueError):
            travel = float("nan")

        scores.append(calculate_reliability_score(label, travel))

    out["reliability_score"] = scores
    return out


if __name__ == "__main__":
    # Tiny example table: all three buffer labels and different trip lengths
    sample = pd.DataFrame(
        {
            "route": ["A", "B", "C", "D"],
            "buffer_label": ["Safe", "Risky", "Very Tight", "Safe"],
            "total_travel_minutes": [20.0, 35.0, 12.0, 100.0],
        }
    )

    scored = add_reliability_score(sample)
    print("Sample routes with reliability_score:\n")
    print(scored.to_string(index=False))
