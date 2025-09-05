# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:25:51 2025

@author: lmendes
"""

# models/autoclave_rate_loss.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


# ---------- helpers ----------

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
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        got = lower.get(name.lower())
        if got is not None:
            return got
    return None


# ---------- 1) NEW: Reactor swap loss — Imposed Daily Rate Method ----------

def compute_reactor_swap_loss_imposed_daily_rate_method(
    inputs: Dict[str, Any],
    operating_schedule: pd.DataFrame
) -> pd.DataFrame:
    """
    Reactor Swap Loss for the *Imposed Daily Rate method*:

      loss(t) = ( imposed_rate_tph - AC_rate_3500 ) * shutdown3500

    Where:
      - AC_rate_3500                : Inputs["AC rates during 3500 maintenance"]
      - imposed_rate_tph            : Inputs["PAL feed imposed rate (t/h)"] if given,
                                      else Inputs["PAL feed imposed daily rate"]/24
      - AC_online, shutdown3500     : from operation_schedule

    Applied only when shutdown3500 > 0 and AC_online > 0.

    We cap the result at **≤ 0** (a loss reduces throughput) so that downstream
    formulas can safely use `PAL_max + loss`.

    Returns (daily):
      - Date
      - Reactor Swap Loss Imposed Daily Rate Method (t)   [≤ 0]
    """
    out_cols = ["Date", "Reactor Swap Loss Imposed Daily Rate Method (t)"]
    if operating_schedule is None or operating_schedule.empty:
        return pd.DataFrame(columns=out_cols)

    # Inputs
    ac_rate_3500 = _get_first_float(inputs, ["AC rates during 3500 maintenance", "AC rate during 3500"], 0.0)
    imposed_rate_tph = _get_first_float(inputs, ["PAL feed imposed rate (t/h)", "PAL feed imposed rate"], None)
    if imposed_rate_tph is None:
        imposed_rate_tph = _get_first_float(inputs, ["PAL feed imposed daily rate"], 0.0)

    # Schedule columns
    date_col = _find_col(operating_schedule, ["Date"])
    sh3500_col = _find_col(operating_schedule, ["3500 shutdown hours"])
    if not all([date_col, sh3500_col]):
        return pd.DataFrame(columns=out_cols)

    sh3500 = pd.to_numeric(operating_schedule[sh3500_col], errors="coerce").fillna(0.0)

    mask = (sh3500 > 0)
    vals = pd.Series(0.0, index=operating_schedule.index)
    vals.loc[mask] = ((ac_rate_3500) - float(imposed_rate_tph)) * sh3500[mask]

    out = pd.DataFrame({
        "Date": operating_schedule[date_col].astype(str),
        # ensure it's a <= 0 loss so it can be added to PAL_max
        "Reactor Swap Loss Imposed Daily Rate Method (t)": vals.clip(upper=0.0),
    })
    return out[out_cols]