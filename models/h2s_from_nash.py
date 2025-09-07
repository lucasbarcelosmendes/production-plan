# models/h2s_from_nash.py
from __future__ import annotations

from typing import Optional, List
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        got = lower.get(str(c).strip().lower())
        if got is not None:
            return got
    return None


def compute_h2s_from_nash_daily(daily_variables: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Consume the preprocessed `daily_variables` dataframe from inputs.py and return:
      - Date
      - Extra H2S from NaSH (t/day)

    Notes:
    - `inputs.py` should have already parsed 'Daily variables' with header on row 2 and
      normalized the `date` column to datetime.
    - This function does not read Excel or manipulate headers.
    """
    cols_out = ["Date", "Extra H2S from NaSH (t/day)"]
    if daily_variables is None or daily_variables.empty:
        return pd.DataFrame(columns=cols_out)

    dv = daily_variables.copy()

    # Locate columns (date is normalized in inputs.py as 'date')
    date_col = _find_col(dv, ["date", "Date"])
    extra_col = _find_col(dv, ["Extra H2S from NaSH (t/day)"])

    if not date_col or not extra_col:
        return pd.DataFrame(columns=cols_out)

    out = dv[[date_col, extra_col]].rename(
        columns={date_col: "Date", extra_col: "Extra H2S from NaSH (t/day)"}
    ).copy()

    # Normalize types
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.loc[out["Date"].notna()].copy()
    out["Extra H2S from NaSH (t/day)"] = pd.to_numeric(
        out["Extra H2S from NaSH (t/day)"], errors="coerce"
    ).fillna(0.0)

    # Match the rest of the pipeline: string Date keys
    out["Date"] = out["Date"].dt.date.astype(str)

    return out
