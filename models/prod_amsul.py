# models/prod_amsul.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


# ---- helpers ----
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


# ---- main ----
def compute_prod_amsul(
    inputs: Dict[str, Any],
    ni_prod_df: pd.DataFrame,
    co_prod_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ammonium Sulfate production per day:

        AmSul (t) = ([Ni Prod (t)] + [Co Prod (t)]) * [AmSul Ratio (based on Ni + Co)]

    Notes:
      - The ratio is used as a direct multiplier in t AmSul per t (Ni+Co).
      - If Ni/Co production or dates are missing, values default to 0 for that day.
    """
    # pull scalar ratio (t AmSul per t Ni+Co)
    ratio = _get_first_float(
        inputs,
        ["AmSul Ratio (based on Ni + Co)", "AmSul Ratio", "Ammonium Sulfate Ratio (Ni+Co)"],
        0.0,
    )

    # pick Ni/Co production columns (tolerant names)
    ni_col = _find_col(ni_prod_df, ["Ni Prod (t)", "Ni production (t)", "Ni Produced (t)"])
    co_col = _find_col(co_prod_df, ["Co Prod (t)", "Co production (t)", "Co Produced (t)"])

    # Build daily base with Date
    dates = set()
    for df in (ni_prod_df, co_prod_df):
        if df is not None and not df.empty and "Date" in df.columns:
            dates.update(df["Date"].dropna().astype(str).unique())
    dates = sorted(dates)
    out = pd.DataFrame({"Date": dates})

    # Merge Ni
    if ni_col:
        tmp = ni_prod_df[["Date", ni_col]].copy()
        tmp["Date"] = tmp["Date"].astype(str)
        tmp.rename(columns={ni_col: "Ni Prod (t)"}, inplace=True)
        tmp["Ni Prod (t)"] = pd.to_numeric(tmp["Ni Prod (t)"], errors="coerce").fillna(0.0)
        out = out.merge(tmp, on="Date", how="left")
    else:
        out["Ni Prod (t)"] = 0.0

    # Merge Co
    if co_col:
        tmp = co_prod_df[["Date", co_col]].copy()
        tmp["Date"] = tmp["Date"].astype(str)
        tmp.rename(columns={co_col: "Co Prod (t)"}, inplace=True)
        tmp["Co Prod (t)"] = pd.to_numeric(tmp["Co Prod (t)"], errors="coerce").fillna(0.0)
        out = out.merge(tmp, on="Date", how="left")
    else:
        out["Co Prod (t)"] = 0.0

    out["Ni Prod (t)"] = pd.to_numeric(out["Ni Prod (t)"], errors="coerce").fillna(0.0)
    out["Co Prod (t)"] = pd.to_numeric(out["Co Prod (t)"], errors="coerce").fillna(0.0)

    # Compute Ammonium Sulfate
    out["AmSul (t)"] = (out["Ni Prod (t)"] + out["Co Prod (t)"]) * float(ratio)

    # Keep only requested output (Date + AmSul)
    return out[["Date", "AmSul (t)"]]
