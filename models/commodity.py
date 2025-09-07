# models/commodity.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


# ----------------- helpers -----------------
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
    """Return sorted union of 'Date' strings across provided dataframes."""
    out = set()
    for df in dfs:
        if isinstance(df, pd.DataFrame) and not df.empty and "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)
            out.update(df["Date"].dropna().unique())
    return sorted(out)


# ----------------- Limestone -----------------
def compute_commodity_limestone(
    inputs: Dict[str, Any],
    mine_plan_daily: pd.DataFrame,   # Date, Fe in PAL Feed (%)
    acid_model_df: pd.DataFrame,     # Date, PAL Feed constrained (t), Sulphuric acid ratio - HPAL autoclaves (t/t)
) -> pd.DataFrame:
    """
    Limestone for neutralization (t) =
        [PAL Feed constrained (t)] *
        ( 2.8 * [Fe in PAL Feed (%)]
          + 44.4 * [HPAL Ni extraction (%)]
          - 744.2 * [MOL Ratio]
          + 446.9 * [Sulphuric acid ratio - HPAL autoclaves (t/t)]
          - 4108.9
        ) / 1000

    Limestone for MOL (t) =
        [Limestone required for MOL] * [MOL Ratio] * [Limestone for neutralization (t)]

    Limestone usage (t) = neutralization + MOL
    """

    # Scalars (use as provided; do NOT rescale percentages)
    hpal_ni_ext = _get_first_float(inputs, ["HPAL Ni extraction (%)", "HPAL Ni Extraction (%)"], 0.0)  # e.g., 0.95
    mol_ratio = _get_first_float(inputs, ["MOL Ratio", "MOL ratio"], 0.0)
    limestone_required_for_mol = _get_first_float(inputs, ["Limestone required for MOL", "Limestone req for MOL"], 0.0)

    dates = _align_dates(mine_plan_daily, acid_model_df)
    out = pd.DataFrame({"Date": dates})

    fe_col = _find_col(mine_plan_daily, ["Fe in PAL Feed (%)", "Fe (%)", "Fe"])
    pf_col = _find_col(acid_model_df, ["PAL Feed constrained (t)", "PAL Feed (t)", "PAL Feed Selected (t)"])
    ratio_col = _find_col(acid_model_df, ["Sulphuric acid ratio - HPAL autoclaves (t/t)"])

    if fe_col:
        t = mine_plan_daily[["Date", fe_col]].copy()
        t.rename(columns={fe_col: "Fe in PAL Feed (%)"}, inplace=True)
        t["Fe in PAL Feed (%)"] = pd.to_numeric(t["Fe in PAL Feed (%)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Fe in PAL Feed (%)"] = 0.0

    if pf_col:
        t = acid_model_df[["Date", pf_col]].copy()
        t.rename(columns={pf_col: "PAL Feed constrained (t)"}, inplace=True)
        t["PAL Feed constrained (t)"] = pd.to_numeric(t["PAL Feed constrained (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["PAL Feed constrained (t)"] = 0.0

    if ratio_col:
        t = acid_model_df[["Date", ratio_col]].copy()
        t.rename(columns={ratio_col: "Sulphuric acid ratio - HPAL autoclaves (t/t)"}, inplace=True)
        t["Sulphuric acid ratio - HPAL autoclaves (t/t)"] = pd.to_numeric(
            t["Sulphuric acid ratio - HPAL autoclaves (t/t)"], errors="coerce"
        ).fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Sulphuric acid ratio - HPAL autoclaves (t/t)"] = 0.0

    pal_t = out["PAL Feed constrained (t)"].astype(float)
    fe_pct = out["Fe in PAL Feed (%)"].astype(float)*100.0
    acid_ratio = out["Sulphuric acid ratio - HPAL autoclaves (t/t)"].astype(float)

    inner = (
        2.8 * fe_pct
        + 44.4 * hpal_ni_ext*100
        - 744.2 * mol_ratio
        + 446.9 * acid_ratio
        - 4108.9
    )

    lime_neutral_t = (pal_t * inner / 1000.0).clip(lower=0.0)
    lime_mol_t = (float(limestone_required_for_mol) * float(mol_ratio) * lime_neutral_t).clip(lower=0.0)
    lime_usage_t = lime_neutral_t + lime_mol_t

    return pd.DataFrame({
        "Date": out["Date"],
        "Limestone for neutralization (t)": lime_neutral_t,
        "Limestone for MOL (t)": lime_mol_t,
        "Limestone usage (t)": lime_usage_t,
    })


# ----------------- Coal -----------------
def compute_commodity_coal(
    inputs: Dict[str, Any],
    operation_schedule: pd.DataFrame,  # Date, Autoclave Operating Hours
    acid_model_df: pd.DataFrame,       # Date, PAL Feed constrained (t), Sulphuric acid produced (t)
    limestone_df: pd.DataFrame,        # Date, Limestone for neutralization (t)
) -> pd.DataFrame:
    """
    Steam to autoclaves (t) = [Steam required for autoclaves (t/h)] * [Autoclave Operating Hours]
    Eletricity (MWh) = 0.043 * [PAL Feed constrained (t)] + 39.17 * 24
    Steam to produce eletricity (t) = [Eletricity (MWh)] * [Steam to produce electricity (t/MWh)]
    Steam to acid plant (t) = [Sulphuric acid produced (t)] * [Steam to acid plant (t/t)]
    Steam to refinery (t) = 13.6 * 2.5 * [Autoclave Operating Hours]
    Steam demand from power plant (t) = autoclaves + electricity + refinery - acid plant
    Coal to power plant (t) = [Steam demand ...] / [Coal to power plant (t/t)]
    Coal to Lime Plant (t) = [Coal to Lime Plant (t/t)] * [Limestone for neutralization (t)] * [MOL Ratio]
    Total coal usage (t) = power + lime
    """
    # Scalars
    steam_req_autoclaves_tph = _get_first_float(inputs, ["Steam required for autoclaves (t/h)"], 0.0)
    steam_per_mwh = _get_first_float(inputs, ["Steam to produce electricity (t/MWh)", "Steam to produce eletricity (t/MWh)"], 0.0)
    steam_to_acid_tpt = _get_first_float(inputs, ["Steam to acid plant (t/t)"], 0.0)
    coal_to_pp_tpt = _get_first_float(inputs, ["Coal to power plant (t/t)"], 0.0)
    coal_to_lime_tpt = _get_first_float(inputs, ["Coal to Lime Plant (t/t)"], 0.0)
    mol_ratio = _get_first_float(inputs, ["MOL Ratio", "MOL ratio"], 0.0)

    dates = _align_dates(operation_schedule, acid_model_df, limestone_df)
    out = pd.DataFrame({"Date": dates})

    # Drivers
    oph_col = _find_col(operation_schedule, ["Autoclave Operating Hours", "Autoclave operating hours", "Operating Hours"])
    pf_col = _find_col(acid_model_df, ["PAL Feed constrained (t)", "PAL Feed (t)"])
    sap_prod_col = _find_col(acid_model_df, ["Sulphuric acid produced (t)"])
    lime_neutral_col = _find_col(limestone_df, ["Limestone for neutralization (t)"])

    if oph_col:
        t = operation_schedule[["Date", oph_col]].copy()
        t.rename(columns={oph_col: "Autoclave Operating Hours"}, inplace=True)
        t["Autoclave Operating Hours"] = pd.to_numeric(t["Autoclave Operating Hours"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Autoclave Operating Hours"] = 0.0

    if pf_col:
        t = acid_model_df[["Date", pf_col]].copy()
        t.rename(columns={pf_col: "PAL Feed constrained (t)"}, inplace=True)
        t["PAL Feed constrained (t)"] = pd.to_numeric(t["PAL Feed constrained (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["PAL Feed constrained (t)"] = 0.0

    if sap_prod_col:
        t = acid_model_df[["Date", sap_prod_col]].copy()
        t.rename(columns={sap_prod_col: "Sulphuric acid produced (t)"}, inplace=True)
        t["Sulphuric acid produced (t)"] = pd.to_numeric(t["Sulphuric acid produced (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Sulphuric acid produced (t)"] = 0.0

    if lime_neutral_col:
        t = limestone_df[["Date", lime_neutral_col]].copy()
        t.rename(columns={lime_neutral_col: "Limestone for neutralization (t)"}, inplace=True)
        t["Limestone for neutralization (t)"] = pd.to_numeric(
            t["Limestone for neutralization (t)"], errors="coerce"
        ).fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Limestone for neutralization (t)"] = 0.0

    # Calcs
    oph = out["Autoclave Operating Hours"].astype(float)
    pal_feed_constr = out["PAL Feed constrained (t)"].astype(float)
    sap_prod_t = out["Sulphuric acid produced (t)"].astype(float)
    lime_neutral_t = out["Limestone for neutralization (t)"].astype(float)

    steam_to_autoclaves_t = steam_req_autoclaves_tph * oph
    eletricity_mwh = 0.043 * pal_feed_constr + 39.17 * 24.0
    steam_to_produce_eletricity_t = (eletricity_mwh * steam_per_mwh) if steam_per_mwh > 0 else pd.Series(0.0, index=out.index, dtype=float)
    steam_to_acid_plant_t = sap_prod_t * steam_to_acid_tpt
    steam_to_refinery_t = 13.6 * 2.5 * oph
    steam_demand_from_pp_t = steam_to_autoclaves_t + steam_to_produce_eletricity_t + steam_to_refinery_t - steam_to_acid_plant_t
    coal_to_power_plant_t = (steam_demand_from_pp_t / coal_to_pp_tpt).clip(lower=0.0) if coal_to_pp_tpt > 0 else pd.Series(0.0, index=out.index, dtype=float)
    coal_to_lime_plant_t = (coal_to_lime_tpt * lime_neutral_t * mol_ratio)
    coal_to_lime_plant_t = pd.to_numeric(coal_to_lime_plant_t, errors="coerce").fillna(0.0).clip(lower=0.0)
    total_coal_usage_t = coal_to_power_plant_t + coal_to_lime_plant_t

    return pd.DataFrame({
        "Date": out["Date"],
        "Steam to autoclaves (t)": steam_to_autoclaves_t,
        "Eletricity (MWh)": eletricity_mwh,
        "Steam to produce eletricity (t)": steam_to_produce_eletricity_t,
        "Steam to acid plant (t)": steam_to_acid_plant_t,
        "Steam to refinery (t)": steam_to_refinery_t,
        "Steam demand from power plant (t)": steam_demand_from_pp_t,
        "Coal to power plant (t)": coal_to_power_plant_t,
        "Coal to Lime Plant (t)": coal_to_lime_plant_t,
        "Total coal usage (t)": total_coal_usage_t,
    })


# ----------------- Sulphur -----------------
def compute_commodity_sulphur(
    inputs: Dict[str, Any],
    acid_model_df: pd.DataFrame,   # needs: Date, Sulphuric acid produced (t), Total H2S used (t), PAL Feed constrained (t)
) -> pd.DataFrame:
    """
    Sulphur for H2SO4 (t) = [Sulphur for H2SO4 (t/t)] * [Sulphuric acid produced (t)]
    Sulphur for H2S   (t) = [Sulphur for H2S (t/t)]   * [Total H2S used (t)]
    Sulphur for ore   (t) = [PAL Feed constrained (t)] * [Sulphur for ore (kg/t)] / 1000
    Total sulphur usage (t) = H2SO4 + H2S + ore
    """
    s_tpt_h2so4 = _get_first_float(inputs, ["Sulphur for H2SO4 (t/t)"], 0.0)
    s_tpt_h2s   = _get_first_float(inputs, ["Sulphur for H2S (t/t)"], 0.0)
    s_kgt_ore   = _get_first_float(inputs, ["Sulphur for ore (kg/t)"], 0.0)

    if not (isinstance(acid_model_df, pd.DataFrame) and not acid_model_df.empty and "Date" in acid_model_df.columns):
        return pd.DataFrame(columns=[
            "Date",
            "Sulphur for H2SO4 (t)",
            "Sulphur for H2S (t)",
            "Sulphur for ore (t)",
            "Total sulphur usage (t)"
        ])

    acid_model_df = acid_model_df.copy()
    acid_model_df["Date"] = acid_model_df["Date"].astype(str)

    # Build a date spine
    out = pd.DataFrame({"Date": sorted(acid_model_df["Date"].dropna().unique())})

    # Pick columns from acid_model_df and merge by Date (avoids index alignment issues)
    sap_prod_col = _find_col(acid_model_df, ["Sulphuric acid produced (t)"])
    h2s_used_col = _find_col(acid_model_df, ["Total H2S used (t)"])
    pal_feed_col = _find_col(acid_model_df, ["PAL Feed constrained (t)", "PAL Feed (t)"])

    cols_map = {}
    if sap_prod_col: cols_map[sap_prod_col] = "Sulphuric acid produced (t)"
    if h2s_used_col: cols_map[h2s_used_col] = "Total H2S used (t)"
    if pal_feed_col: cols_map[pal_feed_col] = "PAL Feed constrained (t)"

    if cols_map:
        tmp = acid_model_df[["Date"] + list(cols_map.keys())].rename(columns=cols_map)
        out = out.merge(tmp, on="Date", how="left")

    # Ensure numeric & fill missing with 0
    for c in ["Sulphuric acid produced (t)", "Total H2S used (t)", "PAL Feed constrained (t)"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # Compute components
    s_h2so4_t = (s_tpt_h2so4 * out["Sulphuric acid produced (t)"]).clip(lower=0.0)
    s_h2s_t   = (s_tpt_h2s   * out["Total H2S used (t)"]).clip(lower=0.0)
    s_ore_t   = (out["PAL Feed constrained (t)"] * (s_kgt_ore / 1000.0)).clip(lower=0.0)
    total_s_t = s_h2so4_t + s_h2s_t + s_ore_t

    return pd.DataFrame({
        "Date": out["Date"],
        "Sulphur for H2SO4 (t)": s_h2so4_t,
        "Sulphur for H2S (t)": s_h2s_t,
        "Sulphur for ore (t)": s_ore_t,
        "Total sulphur usage (t)": total_s_t,
    })


# ----------------- Naphtha -----------------
def compute_commodity_naphtha(
    inputs: Dict[str, Any],
    operation_schedule: pd.DataFrame,  # Date, TPSD shutdown hours
    acid_model_df: pd.DataFrame,       # Date, Total H2S used (t)
    ni_prod_df: pd.DataFrame,          # Date, Ni Prod (t)
    co_prod_df: pd.DataFrame,          # Date, Co Prod (t)
) -> pd.DataFrame:
    """
    Naphtha minimum load (t) = [Naphtha minimum load at 40% capacity (t/h)] * 24 if NOT in TPSD else 0
    Hydrogen for H2S (t) = [Total H2S used (t)] * [Hydrogen for H2S (t/t)]
    Hydrogen for Ni reduction and sintering (t) = [Hydrogen for Ni reduction and sintering (t/t)] * [Ni Prod (t)]
    Hydrogen for Co reduction and sintering (t) = [Hydrogen for Co reduction and sintering (t/t)] * [Co Prod (t)]
    Total hydrogen (t) = H2S + Ni + Co
    Naphtha for hydrogen (t) = [Naphtha for Hydrogen (t/t)] * [Total hydrogen (t)]
    Total naphtha (t) = max(Naphtha for hydrogen (t), Naphtha minimum load (t))
    """
    naph_min_tph = _get_first_float(inputs, ["Naphtha minimum load at 40% capacity (t/h)"], 0.0)
    h2s_h2_tpt   = _get_first_float(inputs, ["Hydrogen for H2S (t/t)"], 0.0)
    ni_h2_tpt    = _get_first_float(inputs, ["Hydrogen for Ni reduction and sintering (t/t)"], 0.0)
    co_h2_tpt    = _get_first_float(inputs, ["Hydrogen for Co reduction and sintering (t/t)"], 0.0)
    naph_for_h2  = _get_first_float(inputs, ["Naphtha for Hydrogen (t/t)"], 0.0)

    dates = _align_dates(operation_schedule, acid_model_df, ni_prod_df, co_prod_df)
    out = pd.DataFrame({"Date": dates})

    tpsd_col = _find_col(operation_schedule, ["TPSD shutdown hours", "TPSD Shutdown Hours"])
    if tpsd_col:
        t = operation_schedule[["Date", tpsd_col]].copy()
        t.rename(columns={tpsd_col: "TPSD shutdown hours"}, inplace=True)
        t["TPSD shutdown hours"] = pd.to_numeric(t["TPSD shutdown hours"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["TPSD shutdown hours"] = 0.0

    h2s_used_col = _find_col(acid_model_df, ["Total H2S used (t)"])
    if h2s_used_col:
        t = acid_model_df[["Date", h2s_used_col]].copy()
        t.rename(columns={h2s_used_col: "Total H2S used (t)"}, inplace=True)
        t["Total H2S used (t)"] = pd.to_numeric(t["Total H2S used (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Total H2S used (t)"] = 0.0

    ni_col = _find_col(ni_prod_df, ["Ni Prod (t)", "Ni production (t)"])
    if ni_col:
        t = ni_prod_df[["Date", ni_col]].copy()
        t.rename(columns={ni_col: "Ni Prod (t)"}, inplace=True)
        t["Ni Prod (t)"] = pd.to_numeric(t["Ni Prod (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Ni Prod (t)"] = 0.0

    co_col = _find_col(co_prod_df, ["Co Prod (t)", "Co production (t)"])
    if co_col:
        t = co_prod_df[["Date", co_col]].copy()
        t.rename(columns={co_col: "Co Prod (t)"}, inplace=True)
        t["Co Prod (t)"] = pd.to_numeric(t["Co Prod (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Co Prod (t)"] = 0.0

    tpsd_active = (out["TPSD shutdown hours"].astype(float) > 0.0).astype(float)
    naphtha_min_load_t = naph_min_tph * 24.0 * (1.0 - tpsd_active)

    h2_for_h2s_t = (out["Total H2S used (t)"] * h2s_h2_tpt).clip(lower=0.0)
    h2_for_ni_t  = (out["Ni Prod (t)"] * ni_h2_tpt).clip(lower=0.0)
    h2_for_co_t  = (out["Co Prod (t)"] * co_h2_tpt).clip(lower=0.0)
    total_h2_t   = h2_for_h2s_t + h2_for_ni_t + h2_for_co_t

    naphtha_for_h2_t = (naph_for_h2 * total_h2_t).clip(lower=0.0)
    total_naphtha_t = pd.concat([naphtha_for_h2_t, naphtha_min_load_t], axis=1).max(axis=1)

    return pd.DataFrame({
        "Date": out["Date"],
        "Naphtha minimum load (t)": naphtha_min_load_t,
        "Hydrogen for H2S (t)": h2_for_h2s_t,
        "Hydrogen for Ni reduction and sintering (t)": h2_for_ni_t,
        "Hydrogen for Co reduction and sintering (t)": h2_for_co_t,
        "Total hydrogen (t)": total_h2_t,
        "Naphtha for hydrogen (t)": naphtha_for_h2_t,
        "Total naphtha (t)": total_naphtha_t,
    })


# ----------------- Ammonia -----------------
def compute_commodity_ammonia(
    inputs: Dict[str, Any],
    operation_schedule: pd.DataFrame,  # Date, TPSD shutdown hours
    acid_model_df: pd.DataFrame,       # Date, PAL Feed constrained (t)
    mine_plan_daily: pd.DataFrame,     # Date, Ni in PAL Feed (%), Co in PAL Feed (%)
) -> pd.DataFrame:
    """
    Ammonia - Utilities 6300 startup (t) = base * 0.5 on first & second day AFTER any TPSD day, else 0.
    Ammonia - Refinery {3900, 4300, 4500} (t) =
        [Ammonia - Refinery {area} (t/t)] *
        ( [PAL Feed constrained (t)] * ([Ni%]*[PAL Ni Rec] + [Co%]*[PAL Co Rec]) / ([Ni in Mixed Sulphide] + [Co in Mixed Sulphide]) )
    Total ammonia (t) = sum of the four components.
    """
    base_startup_t = _get_first_float(inputs, ["Ammonia - Utilities 6300 startup (t)"], 0.0)
    nh3_3900_tpt = _get_first_float(inputs, ["Ammonia - Refinery 3900 (t/t)"], 0.0)
    nh3_4300_tpt = _get_first_float(inputs, ["Ammonia - Refinery 4300 (t/t)"], 0.0)
    nh3_4500_tpt = _get_first_float(inputs, ["Ammonia - Refinery 4500 (t/t)"], 0.0)
    pal_ni_rec = _get_first_float(inputs, ["PAL Ni Recovery"], 0.0)
    pal_co_rec = _get_first_float(inputs, ["PAL Co Recovery"], 0.0)
    ni_ms = _get_first_float(inputs, ["Ni in Mixed Sulphide", "Ni in mixed sulphide"], 0.0)
    co_ms = _get_first_float(inputs, ["Co in Mixed Sulphide", "Co in mixed sulphide"], 0.0)
    denom_ref = ni_ms + co_ms

    dates = _align_dates(operation_schedule, acid_model_df, mine_plan_daily)
    out = pd.DataFrame({"Date": dates})

    # TPSD hours
    tpsd_col = _find_col(operation_schedule, ["TPSD shutdown hours", "TPSD Shutdown Hours"])
    if tpsd_col:
        t = operation_schedule[["Date", tpsd_col]].copy()
        t.rename(columns={tpsd_col: "TPSD shutdown hours"}, inplace=True)
        t["TPSD shutdown hours"] = pd.to_numeric(t["TPSD shutdown hours"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["TPSD shutdown hours"] = 0.0

    # PAL feed constrained
    pf_col = _find_col(acid_model_df, ["PAL Feed constrained (t)", "PAL Feed (t)"])
    if pf_col:
        t = acid_model_df[["Date", pf_col]].copy()
        t.rename(columns={pf_col: "PAL Feed constrained (t)"}, inplace=True)
        t["PAL Feed constrained (t)"] = pd.to_numeric(t["PAL Feed constrained (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["PAL Feed constrained (t)"] = 0.0

    # Ni% / Co%
    ni_col = _find_col(mine_plan_daily, ["Ni in PAL Feed (%)", "Ni (%)", "Ni"])
    co_col = _find_col(mine_plan_daily, ["Co in PAL Feed (%)", "Co (%)", "Co"])

    if ni_col:
        t = mine_plan_daily[["Date", ni_col]].copy()
        t.rename(columns={ni_col: "Ni in PAL Feed (%)"}, inplace=True)
        t["Ni in PAL Feed (%)"] = pd.to_numeric(t["Ni in PAL Feed (%)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Ni in PAL Feed (%)"] = 0.0

    if co_col:
        t = mine_plan_daily[["Date", co_col]].copy()
        t.rename(columns={co_col: "Co in PAL Feed (%)"}, inplace=True)
        t["Co in PAL Feed (%)"] = pd.to_numeric(t["Co in PAL Feed (%)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Co in PAL Feed (%)"] = 0.0

    # Startup logic (first & second calendar day after any TPSD day)
    out = out.sort_values("Date").reset_index(drop=True) # Ensure chronological order if not already
    active = (out["TPSD shutdown hours"].astype(float) > 0.0)
    first_after = (~active) & (active.shift(1, fill_value=False)) # First calendar day after any TPSD day
    second_after = (~active) & (first_after.shift(1, fill_value=False)) # Second calendar day after TPSD
    startup_days = first_after | second_after
    nh3_6300_startup_t = base_startup_t * 0.5 * startup_days.astype(float)

    # Refinery ammonia terms
    pal_feed_t = out["PAL Feed constrained (t)"].astype(float)
    ni_pct = out["Ni in PAL Feed (%)"].astype(float)
    co_pct = out["Co in PAL Feed (%)"].astype(float)

    if denom_ref > 0:
        metal_recovered_t = pal_feed_t * ((ni_pct * pal_ni_rec) + (co_pct * pal_co_rec)) / float(denom_ref)
        metal_recovered_t = metal_recovered_t.clip(lower=0.0)
    else:
        metal_recovered_t = pd.Series(0.0, index=out.index, dtype=float)

    nh3_3900_t = (nh3_3900_tpt * metal_recovered_t).clip(lower=0.0)
    nh3_4300_t = (nh3_4300_tpt * metal_recovered_t).clip(lower=0.0)
    nh3_4500_t = (nh3_4500_tpt * metal_recovered_t).clip(lower=0.0)
    total_nh3_t = nh3_6300_startup_t + nh3_3900_t + nh3_4300_t + nh3_4500_t

    return pd.DataFrame({
        "Date": out["Date"],
        "Ammonia - Utilities 6300 startup (t)": nh3_6300_startup_t,
        "Ammonia - Refinery 3900 (t)": nh3_3900_t,
        "Ammonia - Refinery 4300 (t)": nh3_4300_t,
        "Ammonia - Refinery 4500 (t)": nh3_4500_t,
        "Total ammonia (t)": total_nh3_t,
    })


# ----------------- LPG -----------------
def compute_commodity_lpg(
    inputs: Dict[str, Any],
    operation_schedule: pd.DataFrame,  # Date, SAP shutdown hours
    acid_model_df: pd.DataFrame,       # Date, PAL Feed constrained (t)
    ni_prod_df: pd.DataFrame,          # Date, Ni Prod (t)
    co_prod_df: pd.DataFrame,          # Date, Co Prod (t)
    mixed_sulphide_df: pd.DataFrame,   # Date, Mixed Sulphide Produced (t)  <-- from prod_mixed_sulphide.py
) -> pd.DataFrame:
    """
    Uses Mixed Sulphide Produced (t) from external module.
    Includes: H2S/H2 plant flare, acid plant startup (day after SAP shutdown),
              unplanned acid startups (PAL feed > 0), sulphide area flare, refinery flare,
              ammonia flare, nickel & cobalt furnaces, Total LPG (t).
    """
    # Scalars
    lpg_h2s_h2_flare_t = _get_first_float(inputs, ["LPG for H2S/H2 plant flare (t)", "LPG for H2S / Hydrogen plant flare (t)"], 0.0)
    lpg_acid_startup_t = _get_first_float(inputs, ["LPG for acid plant startup (t)"], 0.0)
    lpg_unplanned_startup_t = _get_first_float(inputs, ["LPG for unplanned acid plant startups (t)"], 0.0)
    lpg_sulphide_flare_t = _get_first_float(inputs, ["LPG for sulphide area flare (t)"], 0.0)
    lpg_refinery_flare_t = _get_first_float(inputs, ["LPG for refinery flare (t)"], 0.0)
    lpg_ammonia_flare_t = _get_first_float(inputs, ["LPG for ammonia flare (t)"], 0.0)
    lpg_nickel_tpt = _get_first_float(inputs, ["LPG for nickel furnace (t/t)"], 0.0)
    lpg_cobalt_tpt = _get_first_float(inputs, ["LPG for cobalt furnace (t/t)"], 0.0)

    dates = _align_dates(operation_schedule, acid_model_df, ni_prod_df, co_prod_df, mixed_sulphide_df)
    out = pd.DataFrame({"Date": dates})

    # SAP shutdown hours
    sap_col = _find_col(operation_schedule, ["SAP shutdown hours", "SAP Shutdown Hours", "UTI - 6700 SAPs"])
    if sap_col:
        t = operation_schedule[["Date", sap_col]].copy()
        t.rename(columns={sap_col: "SAP shutdown hours"}, inplace=True)
        t["SAP shutdown hours"] = pd.to_numeric(t["SAP shutdown hours"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["SAP shutdown hours"] = 0.0

    # PAL feed constrained (for unplanned startups rule)
    pf_col = _find_col(acid_model_df, ["PAL Feed constrained (t)", "PAL Feed (t)"])
    if pf_col:
        t = acid_model_df[["Date", pf_col]].copy()
        t.rename(columns={pf_col: "PAL Feed constrained (t)"}, inplace=True)
        t["PAL Feed constrained (t)"] = pd.to_numeric(t["PAL Feed constrained (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["PAL Feed constrained (t)"] = 0.0

    # Ni / Co production (for furnaces)
    ni_prod_col = _find_col(ni_prod_df, ["Ni Prod (t)", "Ni production (t)"])
    co_prod_col = _find_col(co_prod_df, ["Co Prod (t)", "Co production (t)"])

    if ni_prod_col:
        t = ni_prod_df[["Date", ni_prod_col]].copy()
        t.rename(columns={ni_prod_col: "Ni Prod (t)"}, inplace=True)
        t["Ni Prod (t)"] = pd.to_numeric(t["Ni Prod (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Ni Prod (t)"] = 0.0

    if co_prod_col:
        t = co_prod_df[["Date", co_prod_col]].copy()
        t.rename(columns={co_prod_col: "Co Prod (t)"}, inplace=True)
        t["Co Prod (t)"] = pd.to_numeric(t["Co Prod (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Co Prod (t)"] = 0.0

    # Mixed Sulphide Produced (t) — from external module
    msp_col = _find_col(mixed_sulphide_df, ["Mixed Sulphide Produced (t)"])
    if msp_col:
        t = mixed_sulphide_df[["Date", msp_col]].copy()
        t.rename(columns={msp_col: "Mixed Sulphide Produced (t)"}, inplace=True)
        t["Mixed Sulphide Produced (t)"] = pd.to_numeric(t["Mixed Sulphide Produced (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Mixed Sulphide Produced (t)"] = 0.0

    # Day-after SAP shutdown (restart)
    sap = out["SAP shutdown hours"].astype(float)
    sap_active = (sap > 0.0).astype(int).values
    restart = []
    for i in range(len(sap_active)):
        if i == 0:
            restart.append(False); continue
        restart.append((sap_active[i - 1] == 1) and (sap_active[i] == 0))
    restart = pd.Series(restart, index=out.index)

    pal_feed_t = out["PAL Feed constrained (t)"].astype(float)
    msp_t = out["Mixed Sulphide Produced (t)"].astype(float)

    lpg_h2s_h2_flare = pd.Series(lpg_h2s_h2_flare_t, index=out.index) * (msp_t > 0.0).astype(float)
    lpg_acid_startup = pd.Series(lpg_acid_startup_t, index=out.index) * restart.astype(float)
    lpg_unplanned = pd.Series(lpg_unplanned_startup_t, index=out.index) * (pal_feed_t > 0.0).astype(float)
    lpg_sulphide_flare = pd.Series(lpg_sulphide_flare_t, index=out.index) * (msp_t > 0.0).astype(float)
    lpg_refinery_flare = pd.Series(lpg_refinery_flare_t, index=out.index) * (msp_t > 0.0).astype(float)
    lpg_ammonia_flare = pd.Series(lpg_ammonia_flare_t, index=out.index)

    lpg_nickel_furn = (lpg_nickel_tpt * out["Ni Prod (t)"].astype(float)).clip(lower=0.0)
    lpg_cobalt_furn = (lpg_cobalt_tpt * out["Co Prod (t)"].astype(float)).clip(lower=0.0)

    total_lpg = (
        lpg_h2s_h2_flare
        + lpg_acid_startup
        + lpg_unplanned
        + lpg_sulphide_flare
        + lpg_refinery_flare
        + lpg_ammonia_flare
        + lpg_nickel_furn
        + lpg_cobalt_furn
    ).clip(lower=0.0)

    return pd.DataFrame({
        "Date": out["Date"],
        "LPG for H2S / Hydrogen plant flare (t)": lpg_h2s_h2_flare,
        "LPG for acid plant startup (t)": lpg_acid_startup,
        "LPG for unplanned acid plant startups (t)": lpg_unplanned,
        "LPG for sulphide area flare (t)": lpg_sulphide_flare,
        "LPG for refinery flare (t)": lpg_refinery_flare,
        "LPG for ammonia flare (t)": lpg_ammonia_flare,
        "LPG for nickel furnace (t)": lpg_nickel_furn,
        "LPG for cobalt furnace (t)": lpg_cobalt_furn,
        "Total LPG (t)": total_lpg,
    })


# ----------------- Caustic Soda -----------------
def compute_commodity_caustic_soda(
    inputs: Dict[str, Any],
    mixed_sulphide_df: pd.DataFrame,  # Date, Mixed Sulphide Produced (t)  <-- from prod_mixed_sulphide.py
) -> pd.DataFrame:
    """
    Caustic Soda - 50 wt% basis - 4800 (t) =
        [Caustic Soda - 50 wt% basis - 4800 (t/t)] * [Mixed Sulphide Produced (t)]

    Caustic Soda - 50 wt% basis - 3500 (t) =
        [Caustic Soda - 50 wt% basis - 3500 (t/t)] * [Mixed Sulphide Produced (t)]

    Caustic Soda - 50 wt% basis - Demin water (t) =
        [Caustic Soda - 50 wt% basis - Demin water (t)]   (constant per day)

    Total caustic soda (t) = sum of the three components.
    """
    # Scalars from Inputs
    caustic_4800_tpt = _get_first_float(inputs, ["Caustic Soda - 50 wt% basis - 4800 (t/t)"], 0.0)
    caustic_3500_tpt = _get_first_float(inputs, ["Caustic Soda - 50 wt% basis - 3500 (t/t)"], 0.0)
    caustic_demin_t  = _get_first_float(inputs, ["Caustic Soda - 50 wt% basis - Demin water (t)"], 0.0)

    # Dates skeleton
    dates = _align_dates(mixed_sulphide_df)
    out = pd.DataFrame({"Date": dates})

    # Mixed Sulphide Produced (t)
    msp_col = _find_col(mixed_sulphide_df, ["Mixed Sulphide Produced (t)"])
    if msp_col:
        t = mixed_sulphide_df[["Date", msp_col]].copy()
        t.rename(columns={msp_col: "Mixed Sulphide Produced (t)"}, inplace=True)
        t["Mixed Sulphide Produced (t)"] = pd.to_numeric(t["Mixed Sulphide Produced (t)"], errors="coerce").fillna(0.0)
        out = out.merge(t, on="Date", how="left")
    else:
        out["Mixed Sulphide Produced (t)"] = 0.0

    msp_t = out["Mixed Sulphide Produced (t)"].astype(float)

    # Components
    caustic_4800_t = (caustic_4800_tpt * msp_t).clip(lower=0.0)
    caustic_3500_t = (caustic_3500_tpt * msp_t).clip(lower=0.0)
    caustic_demin_series = pd.Series(caustic_demin_t, index=out.index, dtype=float).clip(lower=0.0)

    total_caustic_t = caustic_4800_t + caustic_3500_t + caustic_demin_series

    return pd.DataFrame({
        "Date": out["Date"],
        "Caustic Soda - 50 wt% basis - 4800 (t)": caustic_4800_t,
        "Caustic Soda - 50 wt% basis - 3500 (t)": caustic_3500_t,
        "Caustic Soda - 50 wt% basis - Demin water (t)": caustic_demin_series,
        "Total caustic soda (t)": total_caustic_t,
    })
