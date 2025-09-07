# models/operation_schedule.py
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd


def _get_first_float(d: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def compute_operation_schedule(
    daily_variables: pd.DataFrame,
    *,
    constants: Dict[str, Any] | None = None,
    total_autoclaves: int = 5,
) -> pd.DataFrame:
    """
    Build the daily plant operation schedule using the pre-processed daily_variables from inputs.py.

    Required columns in `daily_variables` (already normalized by inputs.py):
      - date  (datetime64)
      - standby_autoclaves  (int)
      - hpal_ext_dt_factor  (float; used AS-IS)
      - hpal_int_dt_factor  (float; used AS-IS)
      - tpsd_active, pal_3200_hpal_events, pal_3500_reactor_swaps, pal_3100_thickeners,
        uti_6400_h2s, uti_6700_saps_events, ref_nickel_furnace_events, ref_cobalt_furnace_events

    constants:
      Uses "Hours loss of TPSD RD / RU (per Autoclave)" (or "Hours loss of TPSD RD / RU")
      to compute ramp-loss on the day before and the day after a TPSD.

    Returns per-day columns (same schema as before):
      Date, Calendar hours, 32AU Standby, Autoclave shutdown hours, TPSD shutdown hours,
      Ramp loss hours, Total Planned Downtime, Total Available Production time,
      External Downtime Factor, External Downtime, Internal Downtime Factor, Internal Downtime,
      Autoclave Operating Hours, Autoclaves Online,
      3500 shutdown hours, 3100 shutdown hours, 6400 shutdown hours,
      SAP shutdown hours, Nickel furnace shutdown hours, Cobalt furnace shutdown hours
    """
    constants = constants or {}

    # Defensive rename (in case caller passes slightly different headers)
    rename_map = {}
    for col in daily_variables.columns:
        key = str(col).strip().lower()
        if key == "date":
            rename_map[col] = "date"
        elif ("stand-by" in key or "standby" in key) and "autoclave" in key:
            rename_map[col] = "standby_autoclaves"
        elif "external downtime" in key:
            rename_map[col] = "hpal_ext_dt_factor"
        elif "internal downtime" in key:
            rename_map[col] = "hpal_int_dt_factor"
    dv = daily_variables.rename(columns=rename_map).copy()

    # Ensure types (no scaling of factors; use as-is)
    dv["date"] = pd.to_datetime(dv["date"], errors="coerce")
    dv = dv.loc[dv["date"].notna()].copy()
    dv["standby_autoclaves"] = pd.to_numeric(dv.get("standby_autoclaves", 0), errors="coerce").fillna(0).round().astype(int)
    dv["hpal_ext_dt_factor"] = pd.to_numeric(dv.get("hpal_ext_dt_factor", 0.0), errors="coerce").fillna(0.0)
    dv["hpal_int_dt_factor"] = pd.to_numeric(dv.get("hpal_int_dt_factor", 0.0), errors="coerce").fillna(0.0)

    # Event markers (fill missing with zeros)
    event_cols = [
        "tpsd_active",
        "pal_3200_hpal_events",
        "pal_3500_reactor_swaps",
        "pal_3100_thickeners",
        "uti_6400_h2s",
        "uti_6700_saps_events",
        "ref_nickel_furnace_events",
        "ref_cobalt_furnace_events",
    ]
    for c in event_cols:
        if c not in dv.columns:
            dv[c] = 0
        dv[c] = pd.to_numeric(dv[c], errors="coerce").fillna(0).astype(int)

    dv = dv.sort_values("date").reset_index(drop=True)

    if dv.empty:
        return pd.DataFrame(columns=[
            "Date", "Calendar hours", "32AU Standby", "Autoclave shutdown hours",
            "TPSD shutdown hours", "Ramp loss hours", "Total Planned Downtime",
            "Total Available Production time", "External Downtime Factor", "External Downtime",
            "Internal Downtime Factor", "Internal Downtime", "Autoclave Operating Hours",
            "Autoclaves Online", "3500 shutdown hours", "3100 shutdown hours",
            "6400 shutdown hours", "SAP shutdown hours",
            "Nickel furnace shutdown hours", "Cobalt furnace shutdown hours"
        ])

    calendar_hours_per_day = 24.0 * total_autoclaves

    # TPSD ramp loss per autoclave from constants (applied day-before / day-after TPSD)
    hours_loss = _get_first_float(constants, [
        "Hours loss of TPSD RD / RU (per Autoclave)",
        "Hours loss of TPSD RD / RU",
    ], 0.0)

    # Pre-compute neighbors for TPSD ramp loss
    tpsd_prev = dv["tpsd_active"].shift(1).fillna(0).astype(int)
    tpsd_next = dv["tpsd_active"].shift(-1).fillna(0).astype(int)

    out_rows: List[dict] = []

    for idx, row in dv.iterrows():
        d_date = row["date"].date()
        date_str = d_date.isoformat()

        standby_autoclaves = int(max(row["standby_autoclaves"], 0))
        autoclave_count = max(total_autoclaves - standby_autoclaves, 0)

        # planned components
        standby_hours = 24.0 * standby_autoclaves
        autoclave_shutdown = 0.0
        tpsd_shutdown = 0.0
        ramp_loss = 0.0

        # TPSD day → full plant shutdown, standby doesn't stack
        if row["tpsd_active"] > 0:
            tpsd_shutdown = 24.0 * total_autoclaves
            standby_hours = 0.0
        else:
            # ramp loss on day-before or day-after a TPSD, for ONLINE autoclaves only
            if (tpsd_prev.iloc[idx] > 0) or (tpsd_next.iloc[idx] > 0):
                ramp_loss = hours_loss * autoclave_count

        # PAL 3200 events: 24h per event, and reduce standby (can't be negative)
        if row["pal_3200_hpal_events"] > 0 and tpsd_shutdown == 0.0:
            autoclave_shutdown += 24.0 * row["pal_3200_hpal_events"]
            standby_hours = max(0.0, standby_hours - 24.0 * row["pal_3200_hpal_events"])

        # Other events tracked for reporting only (do not change HPAL operating hours)
        sd3500 = 24.0 * (1 if row["pal_3500_reactor_swaps"] > 0 else 0)
        sd3100 = 24.0 * (1 if row["pal_3100_thickeners"] > 0 else 0)
        sd6400 = 24.0 * (1 if row["uti_6400_h2s"] > 0 else 0)

        # SAPs: 24h per event, cap at 48 (two plants × 24h)
        sd6700_sap = min(48.0, 24.0 * row["uti_6700_saps_events"])

        # Furnaces (reporting only)
        sd_ref_nickel = 24.0 * (1 if row["ref_nickel_furnace_events"] > 0 else 0)
        sd_ref_cobalt = 24.0 * (1 if row["ref_cobalt_furnace_events"] > 0 else 0)

        total_planned_dt = autoclave_shutdown + standby_hours + tpsd_shutdown + ramp_loss
        base_running = max(0.0, calendar_hours_per_day - total_planned_dt)

        f_ext = float(row["hpal_ext_dt_factor"]) if pd.notna(row["hpal_ext_dt_factor"]) else 0.0
        f_int = float(row["hpal_int_dt_factor"]) if pd.notna(row["hpal_int_dt_factor"]) else 0.0

        # External then internal (both over operating hours; factors used AS-IS)
        ext_hours = max(0.0, base_running * f_ext)
        remaining_after_ext = max(0.0, base_running - ext_hours)
        int_hours = max(0.0, remaining_after_ext * f_int)

        operating_hours = max(0.0, base_running - ext_hours - int_hours)
        autoclaves_online = operating_hours / 24.0  # 24 h per autoclave-day

        out_rows.append({
            "Date": date_str,
            "Calendar hours": calendar_hours_per_day,
            "32AU Standby": standby_hours,
            "Autoclave shutdown hours": autoclave_shutdown,
            "TPSD shutdown hours": tpsd_shutdown,
            "Ramp loss hours": ramp_loss,
            "Total Planned Downtime": total_planned_dt,
            "Total Available Production time": base_running,
            "External Downtime Factor": f_ext,
            "External Downtime": ext_hours,
            "Internal Downtime Factor": f_int,
            "Internal Downtime": int_hours,
            "Autoclave Operating Hours": operating_hours,
            "Autoclaves Online": autoclaves_online,
            "3500 shutdown hours": sd3500,
            "3100 shutdown hours": sd3100,
            "6400 shutdown hours": sd6400,
            "SAP shutdown hours": sd6700_sap,
            "Nickel furnace shutdown hours": sd_ref_nickel,
            "Cobalt furnace shutdown hours": sd_ref_cobalt,
        })

    return pd.DataFrame(out_rows)
