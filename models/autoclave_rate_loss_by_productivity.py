# models/autoclave_rate_loss_by_productivity.py
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


def _get_exact_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Exact key lookup (used for 'Maximum autoclave rate')."""
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        got = lower.get(name.lower())
        if got is not None:
            return got
    return None


# ---------- 1) Reactor swap loss (3500) — Productivity Method ----------
def compute_reactor_swap_loss(
    constants: Dict[str, Any],
    operating_schedule: pd.DataFrame
) -> pd.DataFrame:
    """
    Reactor Swap Loss for the *Productivity method*:
      loss = shutdown3500 * ((AC_rate_3500 / AC_online) - AC_rate)
    applied only when shutdown3500 > 0 and AC_online > 0

    Returns (daily):
      - Date
      - Reactor Swap Loss Productivity Method (t)   [negative]
    """
    out_cols = ["Date", "Reactor Swap Loss Productivity Method (t)"]
    if operating_schedule is None or operating_schedule.empty:
        return pd.DataFrame(columns=out_cols)

    # Base AC rate comes from inputs.py constant (exact name)
    ac_rate = _get_exact_float(constants, "Maximum autoclave rate", 0.0)

    # Maintenance rate (keep existing key names; values used as-is)
    ac_rate_3500 = _get_first_float(constants, ["AC rates during 3500 maintenance", "AC rate during 3500"], 0.0)

    # Schedule columns
    date_col = _find_col(operating_schedule, ["Date"])
    sh3500_col = _find_col(operating_schedule, ["3500 shutdown hours"])
    ac_online_col = _find_col(operating_schedule, ["Autoclaves Online"])
    if not all([date_col, sh3500_col, ac_online_col]):
        return pd.DataFrame(columns=out_cols)

    sh3500 = pd.to_numeric(operating_schedule[sh3500_col], errors="coerce").fillna(0.0)
    ac_online = pd.to_numeric(operating_schedule[ac_online_col], errors="coerce").fillna(0.0)

    mask = (sh3500 > 0) & (ac_online > 0)
    loss_vals = pd.Series(0.0, index=operating_schedule.index)
    loss_vals.loc[mask] = sh3500[mask] * ((ac_rate_3500 / ac_online[mask]) - ac_rate)

    out = pd.DataFrame({
        "Date": operating_schedule[date_col].astype(str),
        "Reactor Swap Loss Productivity Method (t)": loss_vals.clip(upper=0.0),
    })
    return out[out_cols]


# ---------- 2) 3100 thickeners loss ----------
def compute_thickeners_shutdown_loss(
    constants: Dict[str, Any],
    operating_schedule: pd.DataFrame
) -> pd.DataFrame:
    """
    Thickeners shutdown loss:
      loss = shutdown3100 * ((AC_rate_3100 / AC_online) - AC_rate)
    applied only when shutdown3100 > 0 and AC_online > 0

    Returns (daily):
      - Date
      - Thickeners shutdown Loss (t)                [negative]
    """
    out_cols = ["Date", "Thickeners shutdown Loss (t)"]
    if operating_schedule is None or operating_schedule.empty:
        return pd.DataFrame(columns=out_cols)

    ac_rate = _get_exact_float(constants, "Maximum autoclave rate", 0.0)
    ac_rate_3100 = _get_first_float(constants, ["AC rates during 3100 maintenance", "AC rate during 3100"], 0.0)

    date_col = _find_col(operating_schedule, ["Date"])
    sh3100_col = _find_col(operating_schedule, ["3100 shutdown hours"])
    ac_online_col = _find_col(operating_schedule, ["Autoclaves Online"])
    if not all([date_col, sh3100_col, ac_online_col]):
        return pd.DataFrame(columns=out_cols)

    sh3100 = pd.to_numeric(operating_schedule[sh3100_col], errors="coerce").fillna(0.0)
    ac_online = pd.to_numeric(operating_schedule[ac_online_col], errors="coerce").fillna(0.0)

    mask = (sh3100 > 0) & (ac_online > 0)
    loss_vals = pd.Series(0.0, index=operating_schedule.index)
    loss_vals.loc[mask] = sh3100[mask] * ((ac_rate_3100 / ac_online[mask]) - ac_rate)

    out = pd.DataFrame({
        "Date": operating_schedule[date_col].astype(str),
        "Thickeners shutdown Loss (t)": loss_vals.clip(upper=0.0),
    })
    return out[out_cols]


# ---------- 3) Other productivity loss ----------
def compute_other_productivity_loss(
    constants: Dict[str, Any],
    pal_feed_max_by_productivity: pd.DataFrame,
    reactor_swap_loss_productivity_method: pd.DataFrame,
    thickeners_shutdown_loss: pd.DataFrame,
) -> pd.DataFrame:
    """
    Other Productivity Loss (t) for the *Productivity method*:

      factor   = 1 - (Internal Productivity Factor * Other External Productivity losses)
      value    = factor * ( PALmax - ReactorSwap - Thickeners )

    Returns (daily):
      - Date
      - Other Productivity Loss (t)                 [negative]
    """
    out_cols = ["Date", "Other Productivity Loss (t)"]
    if pal_feed_max_by_productivity is None or pal_feed_max_by_productivity.empty:
        return pd.DataFrame(columns=out_cols)

    # Factors come from constants (values used as provided)
    internal_factor = _get_first_float(constants, ["Internal Productivity Factor"], 1.0)
    external_factor = _get_first_float(constants, ["Other External Productivity losses"], 1.0)

    # Required columns
    pf = pal_feed_max_by_productivity.copy()
    pf["Date"] = pf["Date"].astype(str)
    pal_col = _find_col(pf, ["PAL Feed max by productivity"])
    if pal_col is None:
        return pd.DataFrame(columns=out_cols)
    pf[pal_col] = pd.to_numeric(pf[pal_col], errors="coerce").fillna(0.0)

    rs = reactor_swap_loss_productivity_method.copy()
    rs["Date"] = rs["Date"].astype(str)
    rs_col = _find_col(rs, ["Reactor Swap Loss Productivity Method (t)"])
    rs[rs_col] = pd.to_numeric(rs[rs_col], errors="coerce").fillna(0.0)

    tk = thickeners_shutdown_loss.copy()
    tk["Date"] = tk["Date"].astype(str)
    tk_col = _find_col(tk, ["Thickeners shutdown Loss (t)"])
    tk[tk_col] = pd.to_numeric(tk[tk_col], errors="coerce").fillna(0.0)

    merged = pf[["Date", pal_col]].merge(rs[["Date", rs_col]], on="Date", how="left") \
                                  .merge(tk[["Date", tk_col]], on="Date", how="left")

    factor = 1.0 - (internal_factor * external_factor)
    merged["Other Productivity Loss (t)"] = -(factor * (merged[pal_col] - merged[rs_col] - merged[tk_col]))

    return merged[out_cols]


# ---------- 4) Convenience wrapper ----------
def compute_autoclave_rate_loss(
    constants: Dict[str, Any],
    operating_schedule: pd.DataFrame,
    pal_feed_max_by_productivity: pd.DataFrame,
) -> pd.DataFrame:
    """Return all 3 results merged by Date."""
    swap  = compute_reactor_swap_loss(constants, operating_schedule)
    thick = compute_thickeners_shutdown_loss(constants, operating_schedule)
    other = compute_other_productivity_loss(constants, pal_feed_max_by_productivity, swap, thick)
    out = swap.merge(thick, on="Date", how="outer").merge(other, on="Date", how="outer")
    return out.sort_values("Date").reset_index(drop=True)
