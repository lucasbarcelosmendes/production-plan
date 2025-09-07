# models/prod_mixed_sulphide.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


# ---------- helpers ----------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        got = lower.get(str(c).strip().lower())
        if got is not None:
            return got
    return None


def _get_first_float(d: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def _align_dates(*dfs: Optional[pd.DataFrame]) -> List[str]:
    """Return sorted union of 'Date' strings across provided DataFrames."""
    out = set()
    for df in dfs:
        if isinstance(df, pd.DataFrame) and not df.empty and "Date" in df.columns:
            out.update(df["Date"].astype(str).dropna().unique())
    return sorted(out)


# ---------- main ----------
def compute_prod_mixed_sulphide(
    inputs: Dict[str, Any],
    acid_model_df: pd.DataFrame,   # needs Date, PAL Feed constrained (t)
    mine_plan_daily: pd.DataFrame, # needs Date, Ni in PAL Feed (%), Co in PAL Feed (%)
) -> pd.DataFrame:
    """
    Computes per-day Mixed Sulphide Produced (t).

    Formula:
      Mixed Sulphide Produced (t) =
          [PAL Feed constrained (t)] *
          ( [Ni in PAL Feed (%)]*[PAL Ni Recovery] + [Co in PAL Feed (%)]*[PAL Co Recovery] )
          / ( [Ni in Mixed Sulphide] + [Co in Mixed Sulphide] )

    Conventions / units:
      • Mine plan Ni/Co are percent values (e.g., 1.25), so we divide by 100.
      • Recoveries are passed as fractions (e.g., 0.93), use as-is.
      • [Ni in Mixed Sulphide] and [Co in Mixed Sulphide] are fractions of the MxS product
        (e.g., 0.54 and 0.05). They are not percents.
      • If the denominator ≤ 0, the output is set to 0 for that day.
    """
    # --- scalars from Inputs ---
    pal_ni_rec = _get_first_float(inputs, ["PAL Ni Recovery"], 0.0)      # fraction, e.g., 0.93
    pal_co_rec = _get_first_float(inputs, ["PAL Co Recovery"], 0.0)      # fraction, e.g., 0.90
    ni_ms = _get_first_float(inputs, ["Ni in Mixed Sulphide", "Ni in mixed sulphide"], 0.0)  # fraction (t/t)
    co_ms = _get_first_float(inputs, ["Co in Mixed Sulphide", "Co in mixed sulphide"], 0.0)  # fraction (t/t)
    denom_ms = ni_ms + co_ms

    # --- build date skeleton ---
    dates = _align_dates(acid_model_df, mine_plan_daily)
    out = pd.DataFrame({"Date": dates})

    # --- source columns ---
    pf_col = _find_col(acid_model_df, ["PAL Feed constrained (t)", "PAL Feed (t)", "PAL Feed Selected (t)"])
    ni_col = _find_col(mine_plan_daily, ["Ni in PAL Feed (%)", "Ni (%)", "Ni"])
    co_col = _find_col(mine_plan_daily, ["Co in PAL Feed (%)", "Co (%)", "Co"])

    # --- merge drivers ---
    if pf_col and "Date" in acid_model_df.columns:
        t = acid_model_df[["Date", pf_col]].copy()
        t.rename(columns={pf_col: "PAL Feed constrained (t)"}, inplace=True)
        t["PAL Feed constrained (t)"] = pd.to_numeric(t["PAL Feed constrained (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["PAL Feed constrained (t)"] = 0.0

    if ni_col and "Date" in mine_plan_daily.columns:
        t = mine_plan_daily[["Date", ni_col]].copy()
        t.rename(columns={ni_col: "Ni in PAL Feed (%)"}, inplace=True)
        t["Ni in PAL Feed (%)"] = pd.to_numeric(t["Ni in PAL Feed (%)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Ni in PAL Feed (%)"] = 0.0

    if co_col and "Date" in mine_plan_daily.columns:
        t = mine_plan_daily[["Date", co_col]].copy()
        t.rename(columns={co_col: "Co in PAL Feed (%)"}, inplace=True)
        t["Co in PAL Feed (%)"] = pd.to_numeric(t["Co in PAL Feed (%)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Co in PAL Feed (%)"] = 0.0

    # --- calculation ---
    pal_feed_t = out["PAL Feed constrained (t)"].astype(float)
    ni_pct = (out["Ni in PAL Feed (%)"].astype(float))
    co_pct = (out["Co in PAL Feed (%)"].astype(float))

    if denom_ms > 0:
        msp_t = pal_feed_t * ((ni_pct * pal_ni_rec) + (co_pct * pal_co_rec)) / float(denom_ms)
        msp_t = msp_t.clip(lower=0.0)
    else:
        msp_t = pd.Series(0.0, index=out.index, dtype=float)

    # --- output ---
    return pd.DataFrame({
        "Date": out["Date"],
        "Mixed Sulphide Produced (t)": msp_t,
    })
