# models/prod_ni.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


# ---------- small helpers (kept local; feel free to move to models/_utils.py) ----------
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



# ---------- main ----------
def compute_prod_co(
    inputs: Dict[str, Any],
    operation_schedule: pd.DataFrame,
    pal_feed_selected: pd.DataFrame,
    mine_plan_daily: pd.DataFrame,          # <-- NEW: Ni% source
) -> pd.DataFrame:
    """
    Per-day Cobalt production & refinery inventory, using:
      Co Fed to Ref  = [Co in PAL Feed (%)] * [PAL Co Recovery] * [PAL Feed (t)]
      Co Prod (t)    = min( [Rate of Co Pack (t/d)],
                            furnace_capacity_t,
                            [Co Fed to Ref] * [REF Recovery % Co] + prev_inventory )
      Co Inventory   = max( prev_inventory + [Co Fed to Ref]*[REF Recovery % Co] - [Co Prod (t)], 0 )

    Furnace capacity factor:
      (24 - [Cobalt furnace shutdown hours]) * [Rate of Cobalt Furnace (t/d)]
    Implemented as tonnes/day equivalent:
      furnace_capacity_t = max(0, 24 - shutdown_hours) / 24.0 * Rate_of_Co_Furnace_tpd

    Scalar inputs taken from 'inputs':
      - "PAL Co Recovery" (fraction or %)   -> treated as fraction
      - "REF Recovery % Co" (fraction or %) -> treated as fraction
      - "Rate of Co Pack (t/d)"
      - "Rate of Cobalt Furnace (t/d)"

    Daily sources:
      - operation_schedule["Cobalt furnace shutdown hours"]
      - pal_feed_selected["PAL Feed (t)"]
      - mine_plan_daily["Co in PAL Feed (%)"] (or label variants)
    """
    # ---- pull scalar inputs ----
    pal_co_rec = _get_first_float(inputs, ["PAL Co Recovery", "PAL Co Recovery (%)"], 0.0)
    ref_co_rec = _get_first_float(inputs, ["REF Recovery % Co", "REF Co Recovery (%)", "Co Recovery % - Refinery"], 0.0)
    pack_rate_tpd = _get_first_float(inputs, ["Rate of Co Pack (t/d)", "Co Pack Rate (t/d)"], 0.0)
    furnace_rate_tpd = _get_first_float(inputs, ["Rate of Cobalt Furnace (t/d)", "Co Furnace Rate (t/d)"], 0.0)
    
    pal_co_rec_f = pal_co_rec
    ref_co_rec_f = ref_co_rec

    # ---- align daily frames by Date ----
    def _pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
        return _find_col(df, names)

    # PAL Feed (t)
    pf_col = _pick_col(pal_feed_selected, ["PAL Feed (t)", "PAL Feed Selected (t)", "PAL Feed final (t)"])
    if pf_col is None:
        return pd.DataFrame(columns=[
            "Date", "Co Fed to Ref (t)", "Co Prod (t)", "Co inventory in refinery (t)", "Cobalt furnace shutdown hours"
        ])
    df_feed = pal_feed_selected[["Date", pf_col]].copy()
    df_feed.rename(columns={pf_col: "PAL Feed (t)"}, inplace=True)
    df_feed["PAL Feed (t)"] = pd.to_numeric(df_feed["PAL Feed (t)"], errors="coerce").fillna(0.0)

    # Cobalt furnace shutdown hours
    sd_col = _pick_col(operation_schedule, ["Cobalt furnace shutdown hours"])
    if sd_col is None:
        df_sd = df_feed[["Date"]].copy()
        df_sd["Cobalt furnace shutdown hours"] = 0.0
    else:
        df_sd = operation_schedule[["Date", sd_col]].copy()
        df_sd.rename(columns={sd_col: "Cobalt furnace shutdown hours"}, inplace=True)
        df_sd["Cobalt furnace shutdown hours"] = pd.to_numeric(df_sd["Cobalt furnace shutdown hours"], errors="coerce").fillna(0.0)

    # Co in PAL Feed (%) from Mine Plan
    co_col = _pick_col(mine_plan_daily, ["Co in PAL Feed (%)", "Co in PAL Feed %", "Co (%)", "Co"])
    if co_col is None:
        # no Co% -> nothing meaningful to compute
        return pd.DataFrame(columns=[
            "Date", "Co Fed to Ref (t)", "Co Prod (t)", "Co inventory in refinery (t)", "Cobalt furnace shutdown hours"
        ])
    df_co = mine_plan_daily[["Date", co_col]].copy()
    df_co.rename(columns={co_col: "Co in PAL Feed (%)"}, inplace=True)
    df_co["Co in PAL Feed (%)"] = pd.to_numeric(df_co["Co in PAL Feed (%)"], errors="coerce").fillna(0.0)

    # Merge by Date
    base = pd.merge(df_feed, df_sd, on="Date", how="left")
    base = pd.merge(base, df_co, on="Date", how="left")
    base["Cobalt furnace shutdown hours"] = base["Cobalt furnace shutdown hours"].fillna(0.0)
    base["Co in PAL Feed (%)"] = base["Co in PAL Feed (%)"].fillna(0.0)

    # ---- compute per-day values ----
    # Co fraction per day
    co_frac_series = base["Co in PAL Feed (%)"]/100

    # Ni Fed to Ref (t)
    base["Co Fed to Ref (t)"] = co_frac_series * pal_co_rec_f * base["PAL Feed (t)"]

    # Furnace capacity (t) using the day’s shutdown hours
    factor_days = (24.0 - base["Cobalt furnace shutdown hours"].clip(lower=0.0)) / 24.0
    factor_days = factor_days.clip(lower=0.0, upper=1.0)
    base["Furnace capacity (t)"] = factor_days * furnace_rate_tpd

    # Rolling Ni Prod & inventory
    prod_list: List[float] = []
    inv_list: List[float] = []
    inv_prev = 0.0  # first day inventory is 0

    for _, row in base.sort_values("Date").iterrows():
        co_fed = float(row["Co Fed to Ref (t)"])
        avail_co = co_fed * ref_co_rec_f + inv_prev
        cap_pack = pack_rate_tpd
        cap_furn = float(row["Furnace capacity (t)"])

        co_prod = min(cap_pack, cap_furn, max(0.0, avail_co))
        inv_now = max(0.0, avail_co - co_prod)

        prod_list.append(co_prod)
        inv_list.append(inv_now)
        inv_prev = inv_now

    out = base[["Date", "Cobalt furnace shutdown hours"]].copy()
    out["Co Fed to Ref (t)"] = base["Co Fed to Ref (t)"]
    out["Co Prod (t)"] = prod_list
    out["Co inventory in refinery (t)"] = inv_list

    # Order & types
    out = out[["Date", "Co Fed to Ref (t)", "Co Prod (t)", "Co inventory in refinery (t)", "Cobalt furnace shutdown hours"]]
    for c in ["Co Fed to Ref (t)", "Co Prod (t)", "Co inventory in refinery (t)", "Cobalt furnace shutdown hours"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out
