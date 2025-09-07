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
def compute_prod_ni(
    inputs: Dict[str, Any],
    operation_schedule: pd.DataFrame,
    pal_feed_selected: pd.DataFrame,
    mine_plan_daily: pd.DataFrame,          # <-- NEW: Ni% source
) -> pd.DataFrame:
    """
    Per-day Nickel production & refinery inventory, using:
      Ni Fed to Ref  = [Ni in PAL Feed (%)] * [PAL Ni Recovery] * [PAL Feed (t)]
      Ni Prod (t)    = min( [Rate of Ni Pack (t/d)],
                            furnace_capacity_t,
                            [Ni Fed to Ref] * [REF Recovery % Ni] + prev_inventory )
      Ni Inventory   = max( prev_inventory + [Ni Fed to Ref]*[REF Recovery % Ni] - [Ni Prod (t)], 0 )

    Furnace capacity factor:
      (2*24 - [Nickel furnace shutdown hours]) * [Rate of Nickel Furnace (t/d)]
    Implemented as tonnes/day equivalent:
      furnace_capacity_t = max(0, 48 - shutdown_hours) / 24.0 * Rate_of_Ni_Furnace_tpd

    Scalar inputs taken from 'inputs':
      - "PAL Ni Recovery" (fraction or %)   -> treated as fraction
      - "REF Recovery % Ni" (fraction or %) -> treated as fraction
      - "Rate of Ni Pack (t/d)"
      - "Rate of Nickel Furnace (t/d)"

    Daily sources:
      - operation_schedule["Nickel furnace shutdown hours"]
      - pal_feed_selected["PAL Feed (t)"]
      - mine_plan_daily["Ni in PAL Feed (%)"] (or label variants)
    """
    # ---- pull scalar inputs ----
    pal_ni_rec = _get_first_float(inputs, ["PAL Ni Recovery", "PAL Ni Recovery (%)"], 0.0)
    ref_ni_rec = _get_first_float(inputs, ["REF Recovery % Ni", "REF Ni Recovery (%)", "Ni Recovery % - Refinery"], 0.0)
    pack_rate_tpd = _get_first_float(inputs, ["Rate of Ni Pack (t/d)", "Ni Pack Rate (t/d)"], 0.0)
    furnace_rate_tpd = _get_first_float(inputs, ["Rate of Nickel Furnace (t/d)", "Ni Furnace Rate (t/d)"], 0.0)
    

    pal_ni_rec_f = pal_ni_rec
    ref_ni_rec_f = ref_ni_rec

    # ---- align daily frames by Date ----
    def _pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
        return _find_col(df, names)

    # PAL Feed (t)
    pf_col = _pick_col(pal_feed_selected, ["PAL Feed (t)", "PAL Feed Selected (t)", "PAL Feed final (t)"])
    if pf_col is None:
        return pd.DataFrame(columns=[
            "Date", "Ni Fed to Ref (t)", "Ni Prod (t)", "Ni inventory in refinery (t)", "Nickel furnace shutdown hours"
        ])
    df_feed = pal_feed_selected[["Date", pf_col]].copy()
    df_feed.rename(columns={pf_col: "PAL Feed (t)"}, inplace=True)
    df_feed["PAL Feed (t)"] = pd.to_numeric(df_feed["PAL Feed (t)"], errors="coerce").fillna(0.0)

    # Nickel furnace shutdown hours
    sd_col = _pick_col(operation_schedule, ["Nickel furnace shutdown hours"])
    if sd_col is None:
        df_sd = df_feed[["Date"]].copy()
        df_sd["Nickel furnace shutdown hours"] = 0.0
    else:
        df_sd = operation_schedule[["Date", sd_col]].copy()
        df_sd.rename(columns={sd_col: "Nickel furnace shutdown hours"}, inplace=True)
        df_sd["Nickel furnace shutdown hours"] = pd.to_numeric(df_sd["Nickel furnace shutdown hours"], errors="coerce").fillna(0.0)

    # Ni in PAL Feed (%) from Mine Plan
    ni_col = _pick_col(mine_plan_daily, ["Ni in PAL Feed (%)", "Ni in PAL Feed %", "Ni (%)", "Ni"])
    if ni_col is None:
        # no Ni% -> nothing meaningful to compute
        return pd.DataFrame(columns=[
            "Date", "Ni Fed to Ref (t)", "Ni Prod (t)", "Ni inventory in refinery (t)", "Nickel furnace shutdown hours"
        ])
    df_ni = mine_plan_daily[["Date", ni_col]].copy()
    df_ni.rename(columns={ni_col: "Ni in PAL Feed (%)"}, inplace=True)
    df_ni["Ni in PAL Feed (%)"] = pd.to_numeric(df_ni["Ni in PAL Feed (%)"], errors="coerce").fillna(0.0)

    # Merge by Date
    base = pd.merge(df_feed, df_sd, on="Date", how="left")
    base = pd.merge(base, df_ni, on="Date", how="left")
    base["Nickel furnace shutdown hours"] = base["Nickel furnace shutdown hours"].fillna(0.0)
    base["Ni in PAL Feed (%)"] = base["Ni in PAL Feed (%)"].fillna(0.0)

    # ---- compute per-day values ----
    # Ni fraction per day
    ni_frac_series = base["Ni in PAL Feed (%)"]

    # Ni Fed to Ref (t)
    base["Ni Fed to Ref (t)"] = ni_frac_series * pal_ni_rec_f * base["PAL Feed (t)"]

    # Furnace capacity (t) using the day’s shutdown hours
    factor_days = (48.0 - base["Nickel furnace shutdown hours"].clip(lower=0.0)) / 24.0
    factor_days = factor_days.clip(lower=0.0, upper=2.0)
    base["Furnace capacity (t)"] = factor_days * furnace_rate_tpd

    # Rolling Ni Prod & inventory
    prod_list: List[float] = []
    inv_list: List[float] = []
    inv_prev = 0.0  # first day inventory is 0

    for _, row in base.sort_values("Date").iterrows():
        ni_fed = float(row["Ni Fed to Ref (t)"])
        avail_ni = ni_fed * ref_ni_rec_f + inv_prev
        cap_pack = pack_rate_tpd
        cap_furn = float(row["Furnace capacity (t)"])

        ni_prod = min(cap_pack, cap_furn, max(0.0, avail_ni))
        inv_now = max(0.0, avail_ni - ni_prod)

        prod_list.append(ni_prod)
        inv_list.append(inv_now)
        inv_prev = inv_now

    out = base[["Date", "Nickel furnace shutdown hours"]].copy()
    out["Ni Fed to Ref (t)"] = base["Ni Fed to Ref (t)"]
    out["Ni Prod (t)"] = prod_list
    out["Ni inventory in refinery (t)"] = inv_list

    # Order & types
    out = out[["Date", "Ni Fed to Ref (t)", "Ni Prod (t)", "Ni inventory in refinery (t)", "Nickel furnace shutdown hours"]]
    for c in ["Ni Fed to Ref (t)", "Ni Prod (t)", "Ni inventory in refinery (t)", "Nickel furnace shutdown hours"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out
