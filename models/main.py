# main.py
import os
from pathlib import Path
import pandas as pd

from models.inputs import load_inputs
from models.operation_schedule import compute_operation_schedule
from models.max_pal_feed_by_productivity import compute_max_pal_feed_by_productivity
from models.pal_feed_by_productivity import compute_pal_feed_by_productivity_final
from models.autoclave_rate_loss_by_productivity import (
    compute_reactor_swap_loss,
    compute_thickeners_shutdown_loss,
    compute_other_productivity_loss,
)
from models.mine_plan import compute_mine_plan_daily_calendar
from models.pal_feed_by_imposed_daily_rate import (
    compute_pal_feed_by_imposed_daily_rate,
)
from models.autoclave_rate_loss_by_imposed_daily_rate import (
    compute_reactor_swap_loss_imposed_daily_rate_method,
)
from models.pal_feed_method_selection import compute_selected_pal_feed
from models.prod_ni import compute_prod_ni
from models.prod_co import compute_prod_co
from models.prod_amsul import compute_prod_amsul
from models.acid_model import compute_acid_model  # consolidated: acid + H2S + inventories (+ constrained PAL)
from models.h2s_from_nash import compute_h2s_from_nash_daily  # NEW

# NEW: mixed sulphide + commodities
from models.prod_mixed_sulphide import compute_prod_mixed_sulphide
from models.commodity import (
    compute_commodity_limestone,
    compute_commodity_coal,
    compute_commodity_sulphur,
    compute_commodity_naphtha,
    compute_commodity_ammonia,
    compute_commodity_lpg,
    compute_commodity_caustic_soda,
)


def _pick_sheet(sheets, *names):
    for n in names:
        if n in sheets:
            return sheets[n]
    lower_map = {str(k).lower(): k for k in sheets}
    for n in names:
        k = lower_map.get(str(n).lower())
        if k is not None:
            return sheets[k]
    return None


def main():
    USER = os.getlogin()

    ROOT = Path(
        fr"C:\Users\{USER}\Ambatovy\DMSA - Asset Performance & Excellence - Documents"
        r"\Strategy and Standardization\3. Improvement Projects\9. Smart Production Plan"
        r"\SPP development\Python"
    )
    INPUT_XLSX = Path(
        fr"C:\Users\{USER}\Ambatovy\DMSA - Asset Performance & Excellence - Documents"
        r"\Strategy and Standardization\3. Improvement Projects\9. Smart Production Plan"
        r"\SPP development\SPP.xlsx"
    )
    OUTPUT_DIR = ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_XLSX = OUTPUT_DIR / "SPP_results.xlsx"

    # Returns cleaned `constants` and `daily_variables` (enriched with shutdown markers)
    constants_new, daily_variables = load_inputs(INPUT_XLSX)

    # 0) NaSH daily (preprocessed daily_variables)
    nash_daily_calendar = compute_h2s_from_nash_daily(daily_variables)

    # 1) Operation Schedule — NEW STRATEGY (consumes inputs.py outputs)
    op_sched = compute_operation_schedule(
        daily_variables=daily_variables,
        constants=constants_new,
        total_autoclaves=5,
    )

    # 2) PAL Feed MAX (by productivity) — unchanged
    pal_feed_max = compute_max_pal_feed_by_productivity(constants_new, op_sched)

    # 3) Losses (productivity method) — now use constants_new from inputs.py
    swap_loss  = compute_reactor_swap_loss(constants_new, op_sched)
    thick_loss = compute_thickeners_shutdown_loss(constants_new, op_sched)
    other_loss = compute_other_productivity_loss(constants_new, pal_feed_max, swap_loss, thick_loss)


    # 4) Mine Plan -> daily Ni/Co% & chemistry drivers — unchanged
    mine_plan_daily = compute_mine_plan_daily_calendar(daily_variables)

    # 5) PAL feed by productivity (net of losses) — unchanged
    pal_feed_final = compute_pal_feed_by_productivity_final(
        pal_feed_by_productivity=pal_feed_max,
        reactor_swap_loss=swap_loss,
        thickeners_shutdown_loss=thick_loss,
        other_productivity_loss=other_loss,
    )

    # 6) Losses (imposed daily rate method) — unchanged
    swap_loss_daily_rate = compute_reactor_swap_loss_imposed_daily_rate_method(constants_new, op_sched)

    # 7) PAL feed by imposed daily rate — unchanged
    pal_feed_imposed = compute_pal_feed_by_imposed_daily_rate(
        inputs=constants_new,
        operation_schedule=op_sched,
        reactor_swap_loss=swap_loss_daily_rate,
    )

    # 8) PAL feed method selection (final selected PAL feed) — unchanged
    pal_feed_selected = compute_selected_pal_feed(
        inputs=constants_new,
        pal_by_productivity=pal_feed_final,
        pal_by_imposed=pal_feed_imposed,
        mine_plan=mine_plan_daily,
        operation_schedule=op_sched,   # for Autoclave Operating Hours -> TOPH
    )

    # ---------- ACID MODEL FIRST (constrains PAL feed and computes acid/H2S/inventories) ----------
    acid_model_df = compute_acid_model(
        inputs=constants_new,
        mine_plan_daily=mine_plan_daily,
        pal_feed_selected=pal_feed_selected,
        operation_schedule=op_sched,
        nash_daily_calendar=nash_daily_calendar,
    )

    # Build a PAL feed DataFrame for production modules using the constrained feed from the acid model — unchanged
    pal_feed_constrained = pd.DataFrame({"Date": acid_model_df.get("Date", pd.Series(dtype=str))})
    if "PAL Feed constrained (t)" in acid_model_df.columns:
        pal_feed_constrained["PAL Feed (t)"] = pd.to_numeric(
            acid_model_df["PAL Feed constrained (t)"], errors="coerce"
        ).fillna(0.0)
    else:
        pf_col = next((c for c in pal_feed_selected.columns if c.lower() == "pal feed (t)".lower()), None)
        pal_feed_constrained = (
            pal_feed_selected[["Date", pf_col]].rename(columns={pf_col: "PAL Feed (t)"})
            if pf_col else pal_feed_constrained
        )

    # 9) Nickel production & refinery inventory — unchanged
    ni_prod_df = compute_prod_ni(constants_new, op_sched, pal_feed_constrained, mine_plan_daily)

    # 10) Cobalt production & refinery inventory — unchanged
    co_prod_df = compute_prod_co(constants_new, op_sched, pal_feed_constrained, mine_plan_daily)

    # 11) Ammonium Sulfate production — unchanged
    amsul_df = compute_prod_amsul(constants_new, ni_prod_df, co_prod_df)

    # ---------- NEW: Mixed Sulphide Production (for commodities) ----------
    mixed_sulphide_df = compute_prod_mixed_sulphide(
        inputs=constants_new,
        acid_model_df=acid_model_df,
        mine_plan_daily=mine_plan_daily,
    )

    # ---------- NEW: Commodities ----------
    limestone_df = compute_commodity_limestone(
        inputs=constants_new,
        mine_plan_daily=mine_plan_daily,
        acid_model_df=acid_model_df,
    )

    coal_df = compute_commodity_coal(
        inputs=constants_new,
        operation_schedule=op_sched,
        acid_model_df=acid_model_df,
        limestone_df=limestone_df,
    )

    sulphur_df = compute_commodity_sulphur(
        inputs=constants_new,
        acid_model_df=acid_model_df,
    )

    naphtha_df = compute_commodity_naphtha(
        inputs=constants_new,
        operation_schedule=op_sched,
        acid_model_df=acid_model_df,
        ni_prod_df=ni_prod_df,
        co_prod_df=co_prod_df,
    )

    ammonia_df = compute_commodity_ammonia(
        inputs=constants_new,
        operation_schedule=op_sched,
        acid_model_df=acid_model_df,
        mine_plan_daily=mine_plan_daily,
    )

    lpg_df = compute_commodity_lpg(
        inputs=constants_new,
        operation_schedule=op_sched,
        acid_model_df=acid_model_df,
        ni_prod_df=ni_prod_df,
        co_prod_df=co_prod_df,
        mixed_sulphide_df=mixed_sulphide_df,
    )

    caustic_df = compute_commodity_caustic_soda(
        inputs=constants_new,
        mixed_sulphide_df=mixed_sulphide_df,
    )

    # ---------- Normalize Date keys ----------
    for df in (
        op_sched, pal_feed_max, swap_loss, thick_loss, other_loss,
        pal_feed_final, pal_feed_imposed, pal_feed_selected,
        mine_plan_daily, ni_prod_df, co_prod_df, amsul_df, acid_model_df,
        pal_feed_constrained, mixed_sulphide_df,
        limestone_df, coal_df, sulphur_df, naphtha_df, ammonia_df, lpg_df, caustic_df
    ):
        if df is not None and not df.empty and "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)

    # ---------- Build final table (merge by Date) ----------
    for col in [
        "Autoclave Rate (t/h) by Productivity",
        "PAL Feed max by productivity",
        "Reactor Swap Loss Productivity Method (t)",
        "Thickeners shutdown Loss (t)",
        "Other Productivity Loss (t)",
        "PAL feed by productivity (t)",
        "PAL feed imposed rate (t/h)",
        "PAL feed maximum rate (t)",
        "Reactor Swap Loss Imposed Daily Rate Method (t)",
        "PAL feed by imposed daily rate (t)",
        "PAL Feed Method",
        "PAL Feed (t)",
        # Products
        "Ni Fed to Ref (t)", "Ni Prod (t)", "Ni inventory in refinery (t)",
        "Co Fed to Ref (t)", "Co Prod (t)", "Co inventory in refinery (t)",
        # AmSul
        "AmSul (t)",
        # Acid/H2S fields that we'll re-merge in the exact order below
        "Sulphuric acid ratio - HPAL autoclaves (t/t)",
        "Sulphuric acid used - HPAL autoclaves (t)",
        "Sulphuric acid used - Demin plant (t)",
        "Sulphuric acid used - Refinery SX (t)",
        "Sulphuric acid used - Ferric (t)",
        "Sulphuric acid used (t)",
        "Nominal Sulphuric Acid produced (t)",
        "Sulphuric acid produced (t)",
        "Sulphuric acid inventory start (t)",
        "Sulphuric acid inventory end (t)",
        "H2S used in Reducing Leach (t)",
        "H2S used in Ref (t)",
        "H2S Used in MxS (t)",
        "Total H2S used (t)",
        "Nominal H2S produced (t)",
        "H2S constraint (t)",
        "PAL Feed constrained (t)",
    ]:
        if col in op_sched.columns:
            op_sched = op_sched.drop(columns=[col])

    merged = op_sched.merge(
        pal_feed_max[["Date", "Autoclave Rate (t/h) by Productivity", "PAL Feed max by productivity"]],
        on="Date", how="left"
    )
    if not swap_loss.empty:
        merged = merged.merge(swap_loss, on="Date", how="left")
    if not thick_loss.empty:
        merged = merged.merge(thick_loss, on="Date", how="left")
    if not other_loss.empty:
        merged = merged.merge(other_loss, on="Date", how="left")
    if pal_feed_final is not None and not pal_feed_final.empty:
        merged = merged.merge(pal_feed_final, on="Date", how="left")
    if pal_feed_imposed is not None and not pal_feed_imposed.empty:
        merged = merged.merge(pal_feed_imposed, on="Date", how="left")
    if pal_feed_selected is not None and not pal_feed_selected.empty:
        merged = merged.merge(pal_feed_selected, on="Date", how="left")

    #Extra H2S from Nash
    if nash_daily_calendar is not None and not nash_daily_calendar.empty:
        merged = merged.merge(nash_daily_calendar, on="Date", how="left")

    # Merge Acid Model in the required order (only include columns that exist)
    if acid_model_df is not None and not acid_model_df.empty:
        acid_cols_desired = [
            "Date",
            "Sulphuric acid used - HPAL autoclaves (t)",
            "Sulphuric acid used - Demin plant (t)",
            "Sulphuric acid used - Refinery SX (t)",
            "Sulphuric acid used - Ferric (t)",
            "Sulphuric acid used (t)",
            "Nominal Sulphuric Acid produced (t)",
            "Sulphuric acid produced (t)",
            "Sulphuric acid inventory start (t)",
            "Sulphuric acid inventory end (t)",
            "H2S used in Reducing Leach (t)",
            "H2S used in Ref (t)",
            "H2S Used in MxS (t)",
            "Total H2S used (t)",
            "Nominal H2S produced (t)",
            "H2S constraint (t)",
            "PAL Feed constrained (t)",
            "TPOH (t/h)"
        ]
        keep = [c for c in acid_cols_desired if c in acid_model_df.columns]
        merged = merged.merge(acid_model_df[keep], on="Date", how="left")

    # Merge Ni/Co/AmSul (computed from constrained PAL)
    if ni_prod_df is not None and not ni_prod_df.empty:
        merged = merged.merge(
            ni_prod_df[["Date", "Ni Fed to Ref (t)", "Ni Prod (t)", "Ni inventory in refinery (t)"]],
            on="Date", how="left"
        )
    if co_prod_df is not None and not co_prod_df.empty:
        merged = merged.merge(
            co_prod_df[["Date", "Co Fed to Ref (t)", "Co Prod (t)", "Co inventory in refinery (t)"]],
            on="Date", how="left"
        )
    if amsul_df is not None and not amsul_df.empty:
        merged = merged.merge(amsul_df, on="Date", how="left")

    # Merge Mixed Sulphide + Commodities
    if mixed_sulphide_df is not None and not mixed_sulphide_df.empty:
        merged = merged.merge(mixed_sulphide_df, on="Date", how="left")
    if limestone_df is not None and not limestone_df.empty:
        merged = merged.merge(limestone_df, on="Date", how="left")
    if coal_df is not None and not coal_df.empty:
        merged = merged.merge(coal_df, on="Date", how="left")
    if sulphur_df is not None and not sulphur_df.empty:
        merged = merged.merge(sulphur_df, on="Date", how="left")
    if naphtha_df is not None and not naphtha_df.empty:
        cols = [c for c in naphtha_df.columns if c != "Mixed Sulphide Produced (t)"]
        merged = merged.merge(naphtha_df[cols], on="Date", how="left")
    if ammonia_df is not None and not ammonia_df.empty:
        merged = merged.merge(ammonia_df, on="Date", how="left")
    if lpg_df is not None and not lpg_df.empty:
        merged = merged.merge(lpg_df, on="Date", how="left")
    if caustic_df is not None and not caustic_df.empty:
        merged = merged.merge(caustic_df, on="Date", how="left")

    # ---------- Monthly summary ----------
    def _monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "Date" not in df.columns:
            return pd.DataFrame(columns=["Month"])
        m = df.copy()
        m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
        m = m[m["Date"].notna()]
        m["Month"] = m["Date"].dt.to_period("M").dt.to_timestamp("M")

        num_cols = m.select_dtypes(include="number").columns.tolist()
        inv_cols = [c for c in num_cols if "inventory" in c.lower()]
        sum_cols = [c for c in num_cols if c not in inv_cols]

        sums = m.groupby("Month", as_index=False)[sum_cols].sum(min_count=1) if sum_cols else m[["Month"]].drop_duplicates()
        inv_last = (
            m.sort_values("Date").groupby("Month")[inv_cols].last().reset_index()
            if inv_cols else sums[["Month"]].copy()
        )
        out = pd.merge(sums, inv_last, on="Month", how="left")
        return out.sort_values("Month").reset_index(drop=True)

    merged_month = _monthly_summary(merged)

    # --- write outputs (multi-tab) ---
    def _write_sheet(xw: pd.ExcelWriter, df: pd.DataFrame, name: str):
        try:
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(xw, sheet_name=name, index=False)
        except Exception:
            pass

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as xw:
        _write_sheet(xw, merged, "Plant Operation Schedule")
        # Node-level tabs
        _write_sheet(xw, op_sched, "Operation Schedule")
        _write_sheet(xw, pal_feed_max, "PAL Max by Prod")
        _write_sheet(xw, swap_loss, "Swap Loss (Prod)")
        _write_sheet(xw, thick_loss, "Thickener Loss")
        _write_sheet(xw, other_loss, "Other Loss (Prod)")
        _write_sheet(xw, pal_feed_final, "PAL by Productivity")
        _write_sheet(xw, pal_feed_imposed, "PAL by Imposed")
        _write_sheet(xw, pal_feed_selected, "PAL Selected")
        _write_sheet(xw, pal_feed_constrained, "PAL Constrained")
        _write_sheet(xw, mine_plan_daily, "Mine Plan Daily")
        _write_sheet(xw, ni_prod_df, "Ni Production")
        _write_sheet(xw, co_prod_df, "Co Production")
        _write_sheet(xw, amsul_df, "AmSul Production")
        _write_sheet(xw, acid_model_df, "Acid Model")
        # NEW tabs
        _write_sheet(xw, mixed_sulphide_df, "Mixed Sulphide Production")
        _write_sheet(xw, limestone_df, "Commodity - Limestone")
        _write_sheet(xw, coal_df, "Commodity - Coal")
        _write_sheet(xw, sulphur_df, "Commodity - Sulphur")
        _write_sheet(xw, naphtha_df, "Commodity - Naphtha")
        _write_sheet(xw, ammonia_df, "Commodity - Ammonia")
        _write_sheet(xw, lpg_df, "Commodity - LPG")
        _write_sheet(xw, caustic_df, "Commodity - Caustic Soda")
        _write_sheet(xw, merged_month, "Summary by Month")

    print(f"[main] Rows written: {len(merged)} -> {OUTPUT_XLSX}")
    return constants_new, mine_plan_daily, merged, merged_month


if __name__ == "__main__":
    constants_new, mine_plan_daily, merged, merged_month = main()
