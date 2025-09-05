# models/pal_feed_method_selection.py
from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd


def _get_first_str(d: Dict[str, Any], keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in d and isinstance(d[k], str):
            return d[k].strip()
    return default


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        got = lower.get(name.lower())
        if got is not None:
            return got
    return None


def compute_selected_pal_feed(
    inputs: Dict[str, Any],
    pal_by_productivity: pd.DataFrame,
    pal_by_imposed: pd.DataFrame,
    mine_plan: pd.DataFrame,
    operation_schedule: Optional[pd.DataFrame] = None,  # <-- NEW (optional)
) -> pd.DataFrame:
    """
    Selects the PAL feed series per the chosen method
    """
    base_cols = ["Date", "PAL Feed Method", "PAL Feed (t)"]  # we'll append TOPH after computing

    method = _get_first_str(inputs, ["PAL Feed Calculation Method"])
    method = method.strip().lower()

    # Normalize all input tables by Date
    tables = [pal_by_productivity, pal_by_imposed, mine_plan, operation_schedule]
    for t in tables:
        if t is not None and not t.empty and "Date" in t.columns:
            t["Date"] = t["Date"].astype(str)

    # Build map of Date -> feed value depending on method
    date_to_value = {}
    method_label = "Unknown"

    if method == "based on productivity":
        method_label = "Based on productivity"
        col = _find_col(pal_by_productivity, ["PAL feed by productivity (t)"])
        if col:
            date_to_value = dict(zip(pal_by_productivity["Date"], pd.to_numeric(pal_by_productivity[col], errors="coerce")))
    elif method == "based on imposed daily rate":
        method_label = "Based on imposed daily rate"
        col = _find_col(pal_by_imposed, ["PAL feed by imposed daily rate (t)"])
        if col:
            date_to_value = dict(zip(pal_by_imposed["Date"], pd.to_numeric(pal_by_imposed[col], errors="coerce")))
    elif method == "mine plan":
        method_label = "Mine Plan"
        col = _find_col(mine_plan, [
            "PAL Feed (t/d)", "PAL Feed (t)", "PAL feed by mine plan (t)",
            "Mine Plan PAL Feed (t/d)", "Mine Plan PAL Feed (t)"
        ])
        if col:
            date_to_value = dict(zip(mine_plan["Date"], pd.to_numeric(mine_plan[col], errors="coerce")))
    # else: Unknown or missing method -> keep empty mapping

    # Determine full date range to output
    all_dates = set()
    for t in [pal_by_productivity, pal_by_imposed, mine_plan]:
        if t is not None and not t.empty and "Date" in t.columns:
            all_dates.update(t["Date"].dropna().unique())
    all_dates = sorted(all_dates)

    out = pd.DataFrame({
        "Date": all_dates,
        "PAL Feed Method": method_label,
        "PAL Feed (t)": [date_to_value.get(d, float("nan")) for d in all_dates],
    })
    out["PAL Feed (t)"] = pd.to_numeric(out["PAL Feed (t)"], errors="coerce")

    # Final column order (TOPH right after PAL Feed)
    out = out[["Date", "PAL Feed Method", "PAL Feed (t)"]]
    return out
