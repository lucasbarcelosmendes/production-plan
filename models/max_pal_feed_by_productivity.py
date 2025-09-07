# models/pal_feed_by_productivity.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    if df is None or df.empty:
        return None
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_map.get(str(name).strip().lower())
        if hit is not None:
            return hit
    return None


def compute_max_pal_feed_by_productivity(inputs: Dict[str, Any], operating_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PAL feed by the productivity method using a constant rate from inputs.

    Inputs expected:
      - inputs["Maximum autoclave rate"]  (t/h)

    operating_schedule must contain:
      - Date
      - Autoclave Operating Hours (any of the common aliases below)

    Returns columns:
      - Date
      - Autoclave Rate (t/h) by Productivity
      - PAL Feed max by productivity  (= Rate * Operating Hours)
    """
    cols = ["Date", "Autoclave Rate (t/h) by Productivity", "PAL Feed max by productivity"]

    if operating_schedule is None or operating_schedule.empty:
        return pd.DataFrame(columns=cols)

    # --- read EXACT constant name (no fallbacks) ---
    try:
        rate = float(inputs.get("Maximum autoclave rate", 0.0))
    except Exception:
        rate = 0.0

    # --- locate required columns in op schedule ---
    date_col = _find_col(operating_schedule, ["Date"])
    op_col = _find_col(
        operating_schedule,
        ["Autoclave Operating Hours", "Operating Hours", "AC Operating Hours"]
    )

    if date_col is None or op_col is None:
        return pd.DataFrame(columns=cols)

    # operating hours per day
    op_hours = pd.to_numeric(operating_schedule[op_col], errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "Date": operating_schedule[date_col].astype(str),
        "Autoclave Rate (t/h) by Productivity": rate,
    })
    out["PAL Feed max by productivity"] = op_hours * rate

    return out[cols]
