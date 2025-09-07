# models/pal_feed_by_imposed_daily_rate.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        hit = lower.get(str(name).strip().lower())
        if hit is not None:
            return hit
    return None


def _get_first_float(d: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def compute_pal_feed_by_imposed_daily_rate(
    inputs: Dict[str, Any],                    # <- now pass constants_new from inputs.py
    operation_schedule: pd.DataFrame,
    reactor_swap_loss: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out_cols = [
        "Date",
        "PAL feed imposed rate (t/h)",
        "PAL feed maximum rate (t)",
        "Reactor Swap Loss Imposed Daily Rate Method (t)",
        "PAL feed by imposed daily rate (t)",
    ]

    if operation_schedule is None or operation_schedule.empty:
        return pd.DataFrame(columns=out_cols)

    # Read imposed rate from constants (inputs.py).
    # Prefer explicit t/h; otherwise accept a daily rate and convert to t/h.
    if "PAL feed imposed rate (t/h)" in inputs:
        try:
            imposed_rate_tph = float(inputs["PAL feed imposed rate (t/h)"])
        except Exception:
            imposed_rate_tph = 0.0
    else:
        # compatibility: support older names if present
        imposed_rate_tph = _get_first_float(inputs, ["PAL feed imposed rate"], None)
        if imposed_rate_tph is None:
            daily_rate_tpd = _get_first_float(inputs, ["PAL feed imposed daily rate"], 0.0)
            imposed_rate_tph = daily_rate_tpd / 24.0

    date_col = _find_col(operation_schedule, ["Date"])
    cal_col = _find_col(operation_schedule, ["Calendar hours"])
    tpsd_col = _find_col(operation_schedule, ["TPSD shutdown hours"])
    if date_col is None or cal_col is None or tpsd_col is None:
        return pd.DataFrame(columns=out_cols)

    sched = operation_schedule[[date_col, cal_col, tpsd_col]].copy()
    sched.columns = ["Date", "Calendar hours", "TPSD shutdown hours"]
    sched["Date"] = sched["Date"].astype(str)

    # If it's a TPSD day, set available hours to 0; otherwise, 24 hours
    sched["TPSD shutdown hours"] = pd.to_numeric(sched["TPSD shutdown hours"], errors="coerce").fillna(0.0)
    available_hours = sched["TPSD shutdown hours"].apply(lambda x: 0.0 if x > 0 else 24.0)

    # Max tonnage at imposed rate
    pal_max_rate_t = available_hours.clip(lower=0.0) * float(imposed_rate_tph)

    out = pd.DataFrame({
        "Date": sched["Date"],
        "PAL feed imposed rate (t/h)": float(imposed_rate_tph),
        "PAL feed maximum rate (t)": pal_max_rate_t,
    })

    # Merge Reactor Swap loss (imposed method), if provided
    rs_map = {}
    if reactor_swap_loss is not None and not reactor_swap_loss.empty:
        dcol = _find_col(reactor_swap_loss, ["Date"])
        vcol = _find_col(reactor_swap_loss, ["Reactor Swap Loss Imposed Daily Rate Method (t)"])
        if dcol and vcol:
            tmp = reactor_swap_loss[[dcol, vcol]].copy()
            tmp[dcol] = tmp[dcol].astype(str)
            rs_map = dict(zip(tmp[dcol], pd.to_numeric(tmp[vcol], errors="coerce").fillna(0.0)))

    out["Reactor Swap Loss Imposed Daily Rate Method (t)"] = out["Date"].map(rs_map).fillna(0.0)

    # Final PAL feed after swap loss
    out["PAL feed by imposed daily rate (t)"] = (
        out["PAL feed maximum rate (t)"] + out["Reactor Swap Loss Imposed Daily Rate Method (t)"]
    )

    return out[out_cols]
