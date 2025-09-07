# models/autoclave_rate_loss_by_imposed_daily_rate.py
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
    if df is None or df.empty:
        return None
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        hit = lower.get(str(name).strip().lower())
        if hit is not None:
            return hit
    return None


# ---------- Reactor swap loss — Imposed Daily Rate Method ----------
def compute_reactor_swap_loss_imposed_daily_rate_method(
    constants: Dict[str, Any],      # <- from inputs.py (constants_new)
    operating_schedule: pd.DataFrame
) -> pd.DataFrame:
    """
    Reactor Swap Loss for the *Imposed Daily Rate method*:

      loss(t) = (AC_rate_3500 - imposed_rate_tph) * shutdown3500
      (returned ≤ 0 so it can be added to PAL_max downstream)

    Inputs (from constants):
      - "AC rates during 3500 maintenance"   (t/h)
      - Either:
          "PAL feed imposed rate (t/h)"      (t/h)  [used as-is], or
          "PAL feed imposed daily rate"      (t/day) → converted to t/h by /24

    From operating_schedule:
      - "3500 shutdown hours"
      - "Autoclaves Online" (if present, loss only when > 0)
    """
    out_cols = ["Date", "Reactor Swap Loss Imposed Daily Rate Method (t)"]
    if operating_schedule is None or operating_schedule.empty:
        return pd.DataFrame(columns=out_cols)

    # constants from inputs.py (values used as provided; only unit conversion t/day -> t/h when needed)
    ac_rate_3500 = _get_first_float(constants, ["AC rates during 3500 maintenance", "AC rate during 3500"], 0.0)

    imposed_rate_tph: Optional[float] = None
    # preferred exact t/h constant
    if "PAL feed imposed rate (t/h)" in constants:
        try:
            imposed_rate_tph = float(constants["PAL feed imposed rate (t/h)"])
        except Exception:
            imposed_rate_tph = 0.0
    else:
        # fallback: daily rate -> convert to t/h
        daily_rate = _get_first_float(constants, ["PAL feed imposed daily rate"], 0.0)
        imposed_rate_tph = daily_rate / 24.0

    # schedule columns
    date_col = _find_col(operating_schedule, ["Date"])
    sh3500_col = _find_col(operating_schedule, ["3500 shutdown hours"])
    ac_online_col = _find_col(operating_schedule, ["Autoclaves Online"])  # optional check

    if not all([date_col, sh3500_col]):
        return pd.DataFrame(columns=out_cols)

    sh3500 = pd.to_numeric(operating_schedule[sh3500_col], errors="coerce").fillna(0.0)
    if ac_online_col:
        ac_online = pd.to_numeric(operating_schedule[ac_online_col], errors="coerce").fillna(0.0)
        mask = (sh3500 > 0) & (ac_online > 0)
    else:
        mask = (sh3500 > 0)

    # loss is negative (≤ 0)
    vals = pd.Series(0.0, index=operating_schedule.index)
    vals.loc[mask] = (ac_rate_3500 - float(imposed_rate_tph)) * sh3500[mask]

    out = pd.DataFrame({
        "Date": operating_schedule[date_col].astype(str),
        "Reactor Swap Loss Imposed Daily Rate Method (t)": vals.clip(upper=0.0),
    })
    return out[out_cols]
