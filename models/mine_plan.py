# models/mine_plan.py
from __future__ import annotations

from typing import Optional, List
import pandas as pd

WANTED_COLS: List[str] = [
    "Ni in PAL Feed (%)",
    "Co in PAL Feed (%)",
    "Fe in PAL Feed (%)",
    "Al in PAL Feed (%)",
    "Mg in PAL Feed (%)",
    "Si in PAL Feed (%)",
    "Mn in PAL Feed (%)",
    "C in PAL Feed (%)",
    "Cr in PAL Feed (%)",
    "Mn:C in PAL Feed (#)",
    "Zn (%)",
    "Cu (%)",
    "ZnCu (#)",
    "Settling (#)",
    "Acid ratio (kg/t)",
]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        hit = lower.get(str(c).strip().lower())
        if hit is not None:
            return hit
    return None


def compute_mine_plan_daily_calendar(daily_variables: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily mine plan *directly from the preprocessed `daily_variables` from inputs.py*.

    Assumptions:
    - `inputs.py` already normalized headers and parsed a 'date' column to datetime.
    - Values are used *as provided* (no % scaling, no fractions).

    Output schema matches the previous compute_mine_plan_daily_calendar:
      Date + the mine-plan chemistry/operability columns (in WANTED_COLS order).
    """
    out_cols = ["Date"] + WANTED_COLS

    if daily_variables is None or daily_variables.empty:
        return pd.DataFrame(columns=out_cols)

    dv = daily_variables.copy()

    # Map/normalize date column from inputs.py ('date' -> 'Date')
    date_col = _find_col(dv, ["date", "Date"])
    if not date_col:
        return pd.DataFrame(columns=out_cols)

    dv["Date"] = pd.to_datetime(dv[date_col], errors="coerce")
    dv = dv.loc[dv["Date"].notna()].copy()
    dv["Date"] = dv["Date"].dt.date.astype(str)

    # Pull only the requested columns (case-insensitive), keep order as WANTED_COLS.
    selected = {"Date": dv["Date"]}
    lower_map = {str(c).strip().lower(): c for c in dv.columns}

    for name in WANTED_COLS:
        src = lower_map.get(name.strip().lower())
        if src is None:
            # Column missing in DV -> create empty numeric col (NaN)
            selected[name] = pd.Series([pd.NA] * len(dv), index=dv.index, dtype="float64")
        else:
            # Coerce to numeric where possible; keep as-is if non-numeric slips through
            selected[name] = pd.to_numeric(dv[src], errors="coerce")

    out = pd.DataFrame(selected, columns=out_cols)
    out = out.sort_values("Date").reset_index(drop=True)
    return out
