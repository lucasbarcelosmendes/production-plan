# models/operation_schedule.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd

EXCEL_EPOCH = datetime(1899, 12, 30)


def _excel_serial_to_date(serial: Any) -> Optional[datetime]:
    if pd.isna(serial):
        return None
    # Accept Timestamp/datetime/serial/string
    if isinstance(serial, (pd.Timestamp, datetime)):
        return pd.to_datetime(serial).to_pydatetime()
    try:
        return EXCEL_EPOCH + timedelta(days=float(serial))
    except Exception:
        try:
            return pd.to_datetime(serial).to_pydatetime()
        except Exception:
            return None


def _to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _get_first_float(d: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        got = lower.get(str(c).lower())
        if got is not None:
            return got
    return None


def compute_operation_schedule(inputs: Dict[str, Any], shutdown_calendar: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Builds the daily plant operation schedule with planned and unplanned hours.

    Returns per-day columns:
      Date, Calendar hours, 32AU Standby, Autoclave shutdown hours, TPSD shutdown hours,
      Ramp loss hours, Total Planned Downtime, Total Available Production time,
      External Downtime Factor, External Downtime, Internal Downtime Factor, Internal Downtime,
      Autoclave Operating Hours, Autoclaves Online,
      3500 shutdown hours, 3100 shutdown hours, 6400 shutdown hours,
      SAP shutdown hours,
      Nickel furnace shutdown hours, Cobalt furnace shutdown hours
    """
    # ---- read assumptions ----
    start_raw = inputs.get("Start date of simulation")
    end_raw = inputs.get("End date of simulation")
    hours_loss = _get_first_float(inputs, [
        "Hours loss of TPSD RD / RU (per Autoclave)",
        "Hours loss of TPSD RD / RU"
    ], 0.0)
    ext_dt = _get_first_float(inputs, ["HPAL External Downtime Factor"], 0.0)  # fraction (e.g., 0.05)
    int_dt = _get_first_float(inputs, ["HPAL Internal Downtime Factor"], 0.0)  # fraction (e.g., 0.05)
    standby_autoclaves = int(_get_first_float(inputs, ["Standby Autoclaves"], 0.0))

    start_dt = _excel_serial_to_date(start_raw)
    end_dt = _excel_serial_to_date(end_raw)
    if start_dt is None or end_dt is None:
        return pd.DataFrame(columns=[
            "Date", "Calendar hours", "32AU Standby", "Autoclave shutdown hours",
            "TPSD shutdown hours", "Ramp loss hours", "Total Planned Downtime",
            "Total Available Production time", "External Downtime Factor", "External Downtime",
            "Internal Downtime Factor", "Internal Downtime", "Autoclave Operating Hours",
            "Autoclaves Online", "3500 shutdown hours", "3100 shutdown hours",
            "6400 shutdown hours", "SAP shutdown hours",
            "Nickel furnace shutdown hours", "Cobalt furnace shutdown hours"
        ])

    total_autoclaves = 5
    autoclave_count = total_autoclaves - max(standby_autoclaves, 0)
    calendar_hours_per_day = 24.0 * total_autoclaves

    # Prepare shutdown calendar
    events: List[tuple] = []
    if shutdown_calendar is not None and not shutdown_calendar.empty:
        sd_col = _find_col(shutdown_calendar, ["Start date", "Start Date", "Start"])
        ed_col = _find_col(shutdown_calendar, ["End date", "End Date", "End"])
        type_col = _find_col(shutdown_calendar, ["Event type", "Type", "Event"])
        id_col = _find_col(shutdown_calendar, [
            "Event ID", "Event Id", "EventID", "ID", "Id"
        ])
        if sd_col and ed_col and type_col:
            for _, r in shutdown_calendar.iterrows():
                s = _excel_serial_to_date(r[sd_col])
                e = _excel_serial_to_date(r[ed_col])
                t = str(r[type_col]).strip() if type_col in r else ""
                eid = str(r[id_col]).strip() if id_col and id_col in r else ""
                if s and e and t:
                    events.append((s.date(), e.date(), t, eid))

    # ---- Pass 1: compute planned losses and base running hours (no ext/int yet) ----
    days = pd.date_range(start_dt.date(), end_dt.date(), freq="D")
    prelim_rows: List[dict] = []

    for d in days:
        d_date = d.date()
        date_str = d_date.isoformat()

        # base planned components
        standby_hours = 24.0 * max(standby_autoclaves, 0)
        autoclave_shutdown = 0.0
        tpsd_shutdown = 0.0
        ramp_loss = 0.0
        tpsd_active = False
        sd3500 = 0.0
        sd3100 = 0.0
        sd6400 = 0.0
        sd6700_sap = 0.0
        sd_ref_nickel = 0.0
        sd_ref_cobalt = 0.0

        # scan events
        for (s_date, e_date, etype, eid) in events:
            # TPSD (affects all autoclaves)
            if etype == "TPSD":
                if s_date <= d_date <= e_date:
                    tpsd_shutdown += 24.0 * (autoclave_count + standby_autoclaves)
                    standby_hours -= 24.0 * standby_autoclaves
                    tpsd_active = True
                else:
                    # ramp loss day-before/day-after
                    if (d_date == (s_date - timedelta(days=1))) or (d_date == (e_date + timedelta(days=1))):
                        ramp_loss += hours_loss * autoclave_count

            # PAL 3200 AUs
            if etype == "PAL - 3200 HPALs" and (s_date <= d_date <= e_date) and (not tpsd_active):
                autoclave_shutdown += 24.0
                standby_hours -= 24.0

            # Reactor swaps 3500
            if etype == "PAL - 3500 Reactor Swaps" and (s_date <= d_date <= e_date) and (not tpsd_active):
                sd3500 = 24.0

            # Thickeners 3100
            if etype == "PAL - 3100 Thickeners" and (s_date <= d_date <= e_date) and (not tpsd_active):
                sd3100 = 24.0

            # 6400 H2S
            if etype == "UTI - 6400 H2S" and (s_date <= d_date <= e_date) and (not tpsd_active):
                sd6400 = 24.0

            # 6700 SAPs → add 24h per event (cap at 48 later)
            if etype == "UTI - 6700 SAPs" and (s_date <= d_date <= e_date):
                sd6700_sap += 24.0

            # NEW: Refinery furnaces (Nickel / Cobalt)
            if etype == "REF - 44/4700 Furnaces" and (s_date <= d_date <= e_date):
                eid_u = eid.upper()
                # Nickel furnace shutdown: 44FR02 OR 44FR01
                if ("44FR02" in eid_u) or ("44FR01" in eid_u):
                    sd_ref_nickel = 24.0
                # Cobalt furnace shutdown: 47FR01
                if ("47FR01" in eid_u):
                    sd_ref_cobalt = 24.0

        # cap SAP shutdown at 48 (two plants × 24h)
        sd6700_sap = min(sd6700_sap, 48.0)

        # totals (refinery shutdowns are tracked but do not affect HPAL operating hours)
        total_planned_dt = autoclave_shutdown + standby_hours + tpsd_shutdown + ramp_loss
        base_running = max(0.0, calendar_hours_per_day - total_planned_dt)  # "Total Available Production time" pre ext/int

        prelim_rows.append({
            "Date": date_str,
            "Calendar hours": calendar_hours_per_day,
            "32AU Standby": standby_hours,
            "Autoclave shutdown hours": autoclave_shutdown,
            "TPSD shutdown hours": tpsd_shutdown,
            "Ramp loss hours": ramp_loss,
            "Total Planned Downtime": total_planned_dt,
            "Total Available Production time": base_running,   # pre ext/int
            "3500 shutdown hours": sd3500,
            "3100 shutdown hours": sd3100,
            "6400 shutdown hours": sd6400,
            "SAP shutdown hours": sd6700_sap,
            "Nickel furnace shutdown hours": sd_ref_nickel,
            "Cobalt furnace shutdown hours": sd_ref_cobalt,
        })

    if not prelim_rows:
        return pd.DataFrame(columns=[
            "Date", "Calendar hours", "32AU Standby", "Autoclave shutdown hours",
            "TPSD shutdown hours", "Ramp loss hours", "Total Planned Downtime",
            "Total Available Production time", "External Downtime Factor", "External Downtime",
            "Internal Downtime Factor", "Internal Downtime", "Autoclave Operating Hours",
            "Autoclaves Online", "3500 shutdown hours", "3100 shutdown hours",
            "6400 shutdown hours", "SAP shutdown hours",
            "Nickel furnace shutdown hours", "Cobalt furnace shutdown hours"
        ])

    # ---- Period totals and proportional allocation of ext/int factors ----
    n_days = len(prelim_rows)
    total_calendar_hours = calendar_hours_per_day * n_days

    # Total downtime hours dictated by factors (as % of calendar time)
    total_ext = max(0.0, ext_dt) * total_calendar_hours
    total_int = max(0.0, int_dt) * total_calendar_hours

    # Proportional weights based on base running hours
    base_list = [r["Total Available Production time"] for r in prelim_rows]
    sum_base = sum(base_list)

    # If there is no base time, set all ext/int to zero
    if sum_base <= 0.0:
        for r in prelim_rows:
            r["External Downtime Factor"] = ext_dt
            r["External Downtime"] = 0.0
            r["Internal Downtime Factor"] = int_dt
            r["Internal Downtime"] = 0.0
            r["Autoclave Operating Hours"] = 0.0
            r["Autoclaves Online"] = 0.0
        return pd.DataFrame(prelim_rows)

    # 1) Allocate External proportionally to base
    ext_alloc = [total_ext * (b / sum_base) for b in base_list]

    # 2) Allocate Internal on the remaining base (post-external), renormalized to avoid negatives
    remaining_after_ext = [max(0.0, b - e) for b, e in zip(base_list, ext_alloc)]
    sum_remaining = sum(remaining_after_ext)

    if total_int > 0.0 and sum_remaining > 0.0:
        int_alloc = [total_int * (rem / sum_remaining) for rem in remaining_after_ext]
    else:
        int_alloc = [0.0 for _ in prelim_rows]

    # Final operating hours and autoclaves online
    out_rows: List[dict] = []
    for r, e, i in zip(prelim_rows, ext_alloc, int_alloc):
        base = r["Total Available Production time"]
        operating_hours = max(0.0, base - e - i)
        # autoclaves online equivalent (per 24h per autoclave)
        denom = (calendar_hours_per_day / total_autoclaves) if calendar_hours_per_day else 0.0
        autoclaves_online = (operating_hours / denom) if denom else 0.0

        out_rows.append({
            "Date": r["Date"],
            "Calendar hours": r["Calendar hours"],
            "32AU Standby": r["32AU Standby"],
            "Autoclave shutdown hours": r["Autoclave shutdown hours"],
            "TPSD shutdown hours": r["TPSD shutdown hours"],
            "Ramp loss hours": r["Ramp loss hours"],
            "Total Planned Downtime": r["Total Planned Downtime"],
            "Total Available Production time": base,  # pre ext/int, as before
            "External Downtime Factor": ext_dt,
            "External Downtime": e,
            "Internal Downtime Factor": int_dt,
            "Internal Downtime": i,
            "Autoclave Operating Hours": operating_hours,
            "Autoclaves Online": autoclaves_online,
            "3500 shutdown hours": r["3500 shutdown hours"],
            "3100 shutdown hours": r["3100 shutdown hours"],
            "6400 shutdown hours": r["6400 shutdown hours"],
            "SAP shutdown hours": r["SAP shutdown hours"],
            "Nickel furnace shutdown hours": r["Nickel furnace shutdown hours"],
            "Cobalt furnace shutdown hours": r["Cobalt furnace shutdown hours"],
        })

    # Sanity: ensure period sums match targets (within floating noise)
    # (We intentionally do not re-adjust here; the two-pass allocation ensures equality.)
    return pd.DataFrame(out_rows)
