# models/pal_feed_by_productivity.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        got = lower.get(str(name).lower())
        if got is not None:
            return got
    return None


def _map_by_date(df: Optional[pd.DataFrame], value_col_candidates: List[str]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    date_col = _find_col(df, ["Date"])
    if date_col is None:
        return {}
    val_col = _find_col(df, value_col_candidates)
    if val_col is None:
        return {}
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        d = str(r[date_col])
        try:
            out[d] = float(r[val_col])
        except Exception:
            out[d] = 0.0
    return out


def compute_pal_feed_by_productivity_final(
    pal_feed_by_productivity: pd.DataFrame,
    reactor_swap_loss: Optional[pd.DataFrame] = None,
    thickeners_shutdown_loss: Optional[pd.DataFrame] = None,
    other_productivity_loss: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Final 'PAL feed by productivity (t)':

        PAL feed by productivity (t)
          = [PAL Feed max by productivity]
          + [Reactor Swap Loss Productivity Method (t)]
          + [Thickeners shutdown Loss (t)]
          + [Other Productivity Loss (t)]

    Notes:
      - H2S and Acid constraints are intentionally EXCLUDED to fully dissociate them
        from this result. They remain separate outputs elsewhere.
    """
    out_cols = ["Date", "PAL feed by productivity (t)"]
    if pal_feed_by_productivity is None or pal_feed_by_productivity.empty:
        return pd.DataFrame(columns=out_cols)

    date_col = _find_col(pal_feed_by_productivity, ["Date"])
    pal_max_col = _find_col(pal_feed_by_productivity, ["PAL Feed max by productivity"])
    if date_col is None or pal_max_col is None:
        return pd.DataFrame(columns=out_cols)

    base = pal_feed_by_productivity[[date_col, pal_max_col]].copy()
    base.columns = ["Date", "PALmax"]
    base["Date"] = base["Date"].astype(str)

    rs  = _map_by_date(reactor_swap_loss, ["Reactor Swap Loss Productivity Method (t)"])
    tk  = _map_by_date(thickeners_shutdown_loss, ["Thickeners shutdown Loss (t)"])
    oth = _map_by_date(other_productivity_loss, ["Other Productivity Loss (t)"])

    def sum_adjust(d: str) -> float:
        return rs.get(d, 0.0) + tk.get(d, 0.0) + oth.get(d, 0.0)

    base["PAL feed by productivity (t)"] = base.apply(
        lambda r: float(r["PALmax"]) + sum_adjust(r["Date"]), axis=1
    )

    return base[["Date", "PAL feed by productivity (t)"]]
