# models/acid_model.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Storage bounds
SAP_STORAGE_CAP = 38000.0
SAP_STORAGE_FLOOR = 8000.0

# H2S constants
GRAMS_PER_TONNE = 1e6
MOLAR_EFFICIENCY = 0.96
SAFETY_FACTOR = 1.1


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


# ---------- main ----------
def compute_acid_model(
    inputs: Dict[str, Any],
    mine_plan_daily: pd.DataFrame,
    pal_feed_selected: pd.DataFrame,
    operation_schedule: Optional[pd.DataFrame] = None,   # for SAP + 6400 shutdown
    reagent_calendar: Optional[pd.DataFrame] = None,     # for extra H2S from NaSH
) -> pd.DataFrame:
    """
    Computes (per day), using PAL-feed-only drivers for acid usage:
      • Sulphuric acid ratio - HPAL autoclaves (t/t)
      • Sulphuric acid used by area (HPAL autoclaves, Demin, Refinery SX, Ferric) (t)
      • Sulphuric acid used (t) [total]
      • Nominal Sulphuric Acid produced (t)
      • Nominal H2S produced (t)
      • Sulphuric acid inventory start/end (t) and actual Sulphuric acid produced (t) [cap-aware]
      • H2S used in Reducing Leach (t), H2S used in Ref (t), H2S Used in MxS (t), Total H2S used (t), H2S constraint (t)
      • PAL Feed constrained (t)  ← final feed after applying H2SO4 floor/cap and H2S availability
    """

    # --- Scalars from Inputs ---
    # HPAL acid ratio inputs (keep existing form)
    solids_pct = _get_first_float(inputs, ["Solids Feed (%)", "Solids feed (%)", "Solids (%)"], 0.0)
    hpal_ni_ext_pct = _get_first_float(inputs, ["HPAL Ni extraction (%)", "HPAL Ni Extraction (%)"], 0.0)

    # Acid t/t coefficients
    demin_tpt  = _get_first_float(inputs, ["Sulphuric acid - Demin plant (t/t)", "Demin plant acid (t/t)", "Demin (t/t)"], 0.0)
    sx_tpt     = _get_first_float(inputs, ["Sulphuric acid - Refinery SX (t/t)", "Refinery SX acid (t/t)", "SX (t/t)"], 0.0)
    ferric_tpt = _get_first_float(inputs, ["Sulphuric acid - Ferric (t/t)", "Ferric acid (t/t)", "Ferric (t/t)"], 0.0)

    # SX processing fraction (0..1): share of metals extracted in PAL that actually drive SX acid today
    sx_processing_frac = _get_first_float(inputs, ["Refinery SX processing fraction", "SX processing fraction"], 1.0)
    sx_processing_frac = float(np.clip(sx_processing_frac, 0.0, 1.0))

    # Nominal Sulphuric Acid production inputs (SAP)
    sap_rate_tph = _get_first_float(inputs, ["SAP Rate (tph)", "SAP Rate", "SAP rate tph"], 0.0)

    # Nominal H2S produced inputs
    cap_per_train = _get_first_float(inputs, ["H2S Capacity per Train (t/day)", "H2S Capacity per Train"], 0.0)
    capacity6400  = _get_first_float(inputs, ["6400 Capacity", "H2S Trains", "6400 capacity"], 0.0)

    # H2S per-ton inputs
    h2s_rl_gpt   = _get_first_float(inputs, ["H2S for RL reduction (g/t)"], 0.0)
    h2s_ref_tpt  = _get_first_float(inputs, ["H2S for Refinery (t/t)"], 0.0)
    pal_ni_rec   = _get_first_float(inputs, ["PAL Ni Recovery"], 0.0)
    pal_co_rec   = _get_first_float(inputs, ["PAL Co Recovery"], 0.0)
    mw_ni        = _get_first_float(inputs, ["MWNi"], 58.6934)
    mw_co        = _get_first_float(inputs, ["MWCo"], 58.9332)
    mw_h2s       = _get_first_float(inputs, ["MWH2S"], 34.0809)
    h2s_mxs_util = _get_first_float(inputs, ["H2S for MXS Precipitation (3500 H2S Utilization)"], 1.0)

    # --- pick/normalize daily sources ---
    # PAL Feed (t)
    pf_col = _find_col(pal_feed_selected, ["PAL Feed (t)", "PAL Feed Selected (t)", "PAL Feed final (t)", "PAL Feed (t/d)"])
    # Mine plan chem for HPAL ratio
    al_col = _find_col(mine_plan_daily, ["Al in PAL Feed (%)", "Al (%)", "Al"])
    mg_col = _find_col(mine_plan_daily, ["Mg in PAL Feed (%)", "Mg (%)", "Mg"])
    fe_col = _find_col(mine_plan_daily, ["Fe in PAL Feed (%)", "Fe (%)", "Fe"])
    # Mine plan Ni/Co grades for SX + H2S
    ni_col = _find_col(mine_plan_daily, ["Ni in PAL Feed (%)", "Ni (%)"])
    co_col = _find_col(mine_plan_daily, ["Co in PAL Feed (%)", "Co (%)"])

    # Date harmonization
    for df in (mine_plan_daily, pal_feed_selected, operation_schedule, reagent_calendar):
        if isinstance(df, pd.DataFrame) and not df.empty and "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)

    # Out dates = union of all available (include op schedule & reagent calendar for nominals)
    all_dates = set()
    for df in (mine_plan_daily, pal_feed_selected, operation_schedule, reagent_calendar):
        if isinstance(df, pd.DataFrame) and not df.empty and "Date" in df.columns:
            all_dates.update(df["Date"].dropna().unique())
    out = pd.DataFrame({"Date": sorted(all_dates)})

    # Merge PAL feed (input)
    if pf_col:
        tmp = pal_feed_selected[["Date", pf_col]].copy()
        tmp.rename(columns={pf_col: "PAL Feed before constraints (t)"}, inplace=True)
        tmp["PAL Feed before constraints (t)"] = pd.to_numeric(tmp["PAL Feed before constraints (t)"], errors="coerce").fillna(0.0)
        out = out.merge(tmp, on="Date", how="left")
    else:
        out["PAL Feed before constraints (t)"] = 0.0

    # Merge mine plan chemistry (%)
    for src_col, dst_name in [
        (al_col, "Al in PAL Feed (%)"),
        (mg_col, "Mg in PAL Feed (%)"),
        (fe_col, "Fe in PAL Feed (%)"),
        (ni_col, "Ni in PAL Feed (%)"),
        (co_col, "Co in PAL Feed (%)"),
    ]:
        if src_col:
            t = mine_plan_daily[["Date", src_col]].copy()
            t.rename(columns={src_col: dst_name}, inplace=True)
            t[dst_name] = pd.to_numeric(t[dst_name], errors="coerce").fillna(0.0)
            out = out.merge(t, on="Date", how="left")
        else:
            out[dst_name] = 0.0

    # Fill NaNs for drivers
    for c in [
        "PAL Feed before constraints (t)",
        "Al in PAL Feed (%)", "Mg in PAL Feed (%)", "Fe in PAL Feed (%)",
        "Ni in PAL Feed (%)", "Co in PAL Feed (%)",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # --- HPAL acid ratio (t per t feed) ---
    out["Sulphuric acid ratio - HPAL autoclaves (t/t)"] = (
        21.3 * out["Al in PAL Feed (%)"]
        + 27.8 * out["Mg in PAL Feed (%)"]
        - 2.4  * out["Fe in PAL Feed (%)"]
        - 7.9  * float(solids_pct) * 100
        + 6.308 * float(hpal_ni_ext_pct) * 100
    ) / 1000.0

    # --- Nominal SAP & H2S production ---
    # Bring SAP & 6400 shutdown hours if operation schedule provided
    if isinstance(operation_schedule, pd.DataFrame) and not operation_schedule.empty:
        date_col = _find_col(operation_schedule, ["Date"])
        sap_sd_col = _find_col(operation_schedule, ["SAP shutdown hours", "SAP Shutdown Hours", "UTI - 6700 SAPs", "SAPs shutdown hours"])
        h2s_sd_col = _find_col(operation_schedule, ["6400 shutdown hours", "6400 Shutdown Hours", "UTI - 6400 H2S"])

        if date_col and sap_sd_col:
            sap = operation_schedule[[date_col, sap_sd_col]].copy()
            sap.rename(columns={date_col: "Date", sap_sd_col: "SAP shutdown hours"}, inplace=True)
            sap["Date"] = sap["Date"].astype(str)
            sap["SAP shutdown hours"] = pd.to_numeric(sap["SAP shutdown hours"], errors="coerce").fillna(0.0).clip(0.0, 48.0)
            out = out.merge(sap, on="Date", how="left")
        else:
            out["SAP shutdown hours"] = 0.0

        if date_col and h2s_sd_col:
            h2s = operation_schedule[[date_col, h2s_sd_col]].copy()
            h2s.rename(columns={date_col: "Date", h2s_sd_col: "6400 shutdown hours"}, inplace=True)
            h2s["Date"] = h2s["Date"].astype(str)
            h2s["6400 shutdown hours"] = pd.to_numeric(h2s["6400 shutdown hours"], errors="coerce").fillna(0.0).clip(lower=0.0)
            out = out.merge(h2s, on="Date", how="left")
        else:
            out["6400 shutdown hours"] = 0.0
    else:
        out["SAP shutdown hours"] = 0.0
        out["6400 shutdown hours"] = 0.0

    # Nominal SAP produced (t)
    out["Nominal Sulphuric Acid produced (t)"] = (48.0 - out["SAP shutdown hours"].fillna(0.0)) * float(sap_rate_tph) / 0.33
    out["Nominal Sulphuric Acid produced (t)"] = out["Nominal Sulphuric Acid produced (t)"].clip(lower=0.0)

    # Nominal H2S produced (t) = (2 - shutdown6400/24) * capPerTrain * capacity6400 + extraH2S
    extra_keys = ["Extra H2S from NaSH (t/day)", "Extra H2S from NaSH", "NaSH H2S (t/day)"]
    extra_by_date: Dict[str, float] = {}
    if isinstance(reagent_calendar, pd.DataFrame) and not reagent_calendar.empty:
        rc_date_col = _find_col(reagent_calendar, ["Date"])
        reagent_col = _find_col(reagent_calendar, extra_keys)
        if rc_date_col and reagent_col:
            rc = reagent_calendar[[rc_date_col, reagent_col]].copy()
            rc["Date"] = rc[rc_date_col].astype(str)
            rc_val = pd.to_numeric(rc[reagent_col], errors="coerce").fillna(0.0)
            extra_by_date = dict(zip(rc["Date"], rc_val))
    extra_series = out["Date"].map(extra_by_date).fillna(0.0)
    shut6400 = pd.to_numeric(out["6400 shutdown hours"], errors="coerce").fillna(0.0)
    out["Nominal H2S produced (t)"] = (2.0 - (shut6400 / 24.0)) * float(cap_per_train) * float(capacity6400) + extra_series
    out["Nominal H2S produced (t)"] = out["Nominal H2S produced (t)"].clip(lower=0.0)

    # --- Coefficients that make acid usage linear in PAL feed ---
    ni_frac = (out["Ni in PAL Feed (%)"] / 100.0).clip(lower=0.0)
    co_frac = (out["Co in PAL Feed (%)"] / 100.0).clip(lower=0.0)

    # Metals per tonne feed effectively driving SX (t metal / t feed)
    metals_per_t_feed = (pal_ni_rec * ni_frac + pal_co_rec * co_frac) * sx_processing_frac

    # Total acid coefficient per tonne feed (t acid / t feed)
    coeff_tpt = (
        out["Sulphuric acid ratio - HPAL autoclaves (t/t)"]  # HPAL
        + float(demin_tpt)                                   # Demin
        + float(ferric_tpt)                                  # Ferric
        + float(sx_tpt) * metals_per_t_feed                  # SX (from feed only)
    ).clip(lower=0.0)

    # --- H2S per-tonne (t H2S / t feed) denominator (depends on feed chemistry & recoveries) ---
    term_RL  = h2s_rl_gpt / GRAMS_PER_TONNE
    term_REF = h2s_ref_tpt * (pal_ni_rec * ni_frac + pal_co_rec * co_frac)
    with np.errstate(divide="ignore", invalid="ignore"):
        moles_per_t = (pal_ni_rec * ni_frac) / (MOLAR_EFFICIENCY * mw_ni) + \
                      (pal_co_rec * co_frac) / (MOLAR_EFFICIENCY * mw_co)
    term_MXS = moles_per_t * SAFETY_FACTOR * mw_h2s / max(h2s_mxs_util, 1e-12)
    h2s_denom_tpt = (term_RL + term_REF + term_MXS).clip(lower=0.0)  # t H2S / t feed

    # --- Forward pass: apply H2SO4 floor/cap and H2S availability in-day ---
    out = out.sort_values("Date").reset_index(drop=True)
    pal_in = out["PAL Feed before constraints (t)"].astype(float)
    p_nom  = out["Nominal Sulphuric Acid produced (t)"].astype(float)
    h2s_nom= out["Nominal H2S produced (t)"].astype(float)

    inv_start_list, inv_end_list, prod_actual_list = [], [], []
    pal_final_list = []
    used_h2s_RL_list, used_h2s_REF_list, used_h2s_MXS_list, used_h2s_total_list, h2s_constraint_list = [], [], [], [], []

    S = SAP_STORAGE_CAP  # first day start inventory = 38,000

    for i in range(len(out)):
        coeff_i = float(max(0.0, coeff_tpt.iloc[i]))
        denom_i = float(max(0.0, h2s_denom_tpt.iloc[i]))
        pal_in_i = float(max(0.0, pal_in.iloc[i]))
        P_nom_i  = float(max(0.0, p_nom.iloc[i]))
        H2S_nom_i= float(max(0.0, h2s_nom.iloc[i]))

        # Acid floor-based feed max
        limit_acid = S + P_nom_i - SAP_STORAGE_FLOOR
        pal_acid_max = 0.0 if (limit_acid <= 0.0 or coeff_i <= 0.0) else (limit_acid / coeff_i)

        # H2S-based feed max (if denom_i == 0 -> no H2S requirement from feed)
        pal_h2s_max = float('inf') if denom_i == 0.0 else (H2S_nom_i / denom_i)

        # Final feed after both constraints
        pal_final = min(pal_in_i, pal_acid_max, pal_h2s_max)
        pal_final = max(0.0, pal_final)

        # Acid usage & inventory
        used_acid = coeff_i * pal_final
        end_unconstrained = S + P_nom_i - used_acid
        if end_unconstrained > SAP_STORAGE_CAP:
            end_inv = SAP_STORAGE_CAP
            prod_act = SAP_STORAGE_CAP - S + used_acid
        else:
            end_inv = end_unconstrained
            prod_act = P_nom_i

        # H2S usage breakdown (based on pal_final)
        h2s_used_total = min(H2S_nom_i, denom_i * pal_final) if denom_i > 0.0 else 0.0
        # split by the three terms
        used_RL  = pal_final * term_RL
        used_REF = pal_final * term_REF.iloc[i]
        used_MXS = pal_final * term_MXS.iloc[i]
        # cap tiny numeric overage
        scale = 1.0
        sum_parts = used_RL + used_REF + used_MXS
        if h2s_used_total > 0 and sum_parts > 0:
            scale = min(1.0, h2s_used_total / sum_parts)
        used_RL *= scale; used_REF *= scale; used_MXS *= scale
        h2s_constraint_t = min(0.0, (H2S_nom_i / denom_i) - pal_final) if denom_i > 0.0 else 0.0

        # Carry
        inv_start_list.append(S)
        inv_end_list.append(max(0.0, end_inv))
        prod_actual_list.append(max(0.0, prod_act))
        pal_final_list.append(pal_final)
        used_h2s_RL_list.append(max(0.0, used_RL))
        used_h2s_REF_list.append(max(0.0, used_REF))
        used_h2s_MXS_list.append(max(0.0, used_MXS))
        used_h2s_total_list.append(max(0.0, min(h2s_used_total, used_RL + used_REF + used_MXS)))
        h2s_constraint_list.append(h2s_constraint_t)

        S = end_inv  # next day start

    # --- Build outputs based on PAL Feed constrained (actual) ---
    out["PAL Feed constrained (t)"] = pal_final_list

    # Recompute acid by area from constrained feed
    pal = out["PAL Feed constrained (t)"].astype(float)

    out["Sulphuric acid used - HPAL autoclaves (t)"] = pal * out["Sulphuric acid ratio - HPAL autoclaves (t/t)"]
    out["Sulphuric acid used - Demin plant (t)"]     = pal * float(demin_tpt)
    out["Sulphuric acid used - Ferric (t)"]          = pal * float(ferric_tpt)
    out["Sulphuric acid used - Refinery SX (t)"]     = pal * (float(sx_tpt) * metals_per_t_feed)
    out["Sulphuric acid used (t)"] = (
        out["Sulphuric acid used - HPAL autoclaves (t)"]
        + out["Sulphuric acid used - Demin plant (t)"]
        + out["Sulphuric acid used - Refinery SX (t)"]
        + out["Sulphuric acid used - Ferric (t)"]
    )

    # Inventory & actual SAP production (from forward pass)
    out["Sulphuric acid inventory start (t)"] = inv_start_list
    out["Sulphuric acid produced (t)"]        = prod_actual_list
    out["Sulphuric acid inventory end (t)"]   = inv_end_list

    # H2S outputs
    out["H2S used in Reducing Leach (t)"] = used_h2s_RL_list
    out["H2S used in Ref (t)"]            = used_h2s_REF_list
    out["H2S Used in MxS (t)"]            = used_h2s_MXS_list
    out["Total H2S used (t)"]             = used_h2s_total_list
    out["H2S constraint (t)"]             = h2s_constraint_list

    # ---- Compute TPOH (t/h) = PAL Feed constrained (t) / Autoclave Operating Hours ----
    if isinstance(operation_schedule, pd.DataFrame) and not operation_schedule.empty:
        date_col  = _find_col(operation_schedule, ["Date"])
        hours_col = _find_col(operation_schedule, [
            "Autoclave Operating Hours",
            "Autoclaves Operating Hours",
            "Operating Hours",
            "Autoclave operating hours",
        ])
        if date_col and hours_col:
            # Normalize dates to string for a safe join
            hrs_map = dict(
                zip(
                    operation_schedule[date_col].astype(str),
                    pd.to_numeric(operation_schedule[hours_col], errors="coerce").fillna(0.0)
                )
            )
            hours_series = out["Date"].map(hrs_map).astype(float).fillna(0.0)
            # avoid divide-by-zero -> 0
            with np.errstate(divide="ignore", invalid="ignore"):
                tpoh = out["PAL Feed constrained (t)"].astype(float) / hours_series.replace(0.0, np.nan)
            tpoh = tpoh.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            tpoh = pd.Series(0.0, index=out.index, dtype=float)
    else:
        tpoh = pd.Series(0.0, index=out.index, dtype=float)

    out["TPOH (t/h)"] = tpoh

    # Tidy columns (you can adjust the order to taste)
    out = out[[
        "Date",
        "PAL Feed before constraints (t)",
        "PAL Feed constrained (t)",
        "TPOH (t/h)",
        # Chemistry drivers
        "Al in PAL Feed (%)", "Mg in PAL Feed (%)", "Fe in PAL Feed (%)",
        "Ni in PAL Feed (%)", "Co in PAL Feed (%)",
        # Acid ratios & uses
        "Sulphuric acid ratio - HPAL autoclaves (t/t)",
        "Sulphuric acid used - HPAL autoclaves (t)",
        "Sulphuric acid used - Demin plant (t)",
        "Sulphuric acid used - Refinery SX (t)",
        "Sulphuric acid used - Ferric (t)",
        "Sulphuric acid used (t)",
        # SAP & H2S
        "Nominal Sulphuric Acid produced (t)",
        "Sulphuric acid produced (t)",
        "Sulphuric acid inventory start (t)",
        "Sulphuric acid inventory end (t)",
        "Nominal H2S produced (t)",
        "H2S used in Reducing Leach (t)",
        "H2S used in Ref (t)",
        "H2S Used in MxS (t)",
        "Total H2S used (t)",
        "H2S constraint (t)",
    ]]

    return out