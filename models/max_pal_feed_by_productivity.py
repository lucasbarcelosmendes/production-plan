# models/pal_feed_by_productivity.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


def _get_first_float(d: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def compute_max_pal_feed_by_productivity(inputs: Dict[str, Any], operating_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PAL feed by the productivity method.

    Returns columns:
      - Date
      - Autoclave Rate (t/h) by Productivity
      - PAL Feed max by productivity  (= Rate * Operating Hours)
    """
    cols = ["Date", "Autoclave Rate (t/h) by Productivity", "PAL Feed max by productivity"]
    if operating_schedule is None or operating_schedule.empty:
        return pd.DataFrame(columns=cols)

    # Rate from inputs (t/h)
    rate = _get_first_float(
        inputs,
        ["Autoclave Rates", "Autoclave Rate (t/h)", "Autoclave Rate", "Autoclave rate tph"],
        0.0,
    )

    # Need operating hours for the calc, but we won't include it in the output
    date_col = _find_col(operating_schedule, ["Date"])
    op_col = _find_col(operating_schedule, ["Autoclave Operating Hours", "Operating Hours", "AC Operating Hours"])

    if date_col is None or op_col is None:
        return pd.DataFrame(columns=cols)

    op_hours = pd.to_numeric(operating_schedule[op_col], errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "Date": operating_schedule[date_col].astype(str),
        "Autoclave Rate (t/h) by Productivity": rate,
    })
    out["PAL Feed max by productivity"] = op_hours * rate
    return out[cols]
