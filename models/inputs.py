# models/inputs.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import pandas as pd


# ---------- small utils ----------
def _find_sheet(book: Dict[str, pd.DataFrame], *names: str) -> Optional[str]:
    if not book:
        return None
    # exact
    for n in names:
        if n in book:
            return n
    # case-insensitive
    lower = {str(k).lower(): k for k in book}
    for n in names:
        k = lower.get(str(n).lower())
        if k is not None:
            return k
    return None


def _parse_constants(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Accepts a 2-column table (Key, Value) or a table with columns 'Key'/'Value'."""
    if df is None or df.empty:
        return {}
    cols_lower = [str(c).lower() for c in df.columns]
    if "key" in cols_lower and "value" in cols_lower:
        key_col = df.columns[cols_lower.index("key")]
        val_col = df.columns[cols_lower.index("value")]
        pairs = zip(df[key_col], df[val_col])
    else:
        if df.shape[1] < 2:
            return {}
        pairs = zip(df.iloc[:, 0], df.iloc[:, 1])

    out: Dict[str, Any] = {}
    for k, v in pairs:
        if pd.isna(k):
            continue
        out[str(k).strip()] = v
    return out


def _excel_to_datetime(x: Any) -> Optional[datetime]:
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        got = lower.get(str(c).lower())
        if got is not None:
            return got
    return None


# ---------- Shutdown Calendar -> per-day markers ----------
def _shutdown_calendar_to_daily(calendar_df: Optional[pd.DataFrame], all_dates: pd.Series) -> pd.DataFrame:
    """
    Convert the event calendar into a per-date table of markers/counts aligned to `all_dates`.
    Returns columns:
      - tpsd_active (0/1)
      - pal_3200_hpal_events, pal_3500_reactor_swaps, pal_3100_thickeners
      - uti_6400_h2s, uti_6700_saps_events
      - ref_nickel_furnace_events, ref_cobalt_furnace_events
    """
    out = pd.DataFrame({"date": pd.to_datetime(all_dates, errors="coerce")}).dropna().drop_duplicates()
    out = out.sort_values("date").reset_index(drop=True)

    for c in [
        "tpsd_active",
        "pal_3200_hpal_events",
        "pal_3500_reactor_swaps",
        "pal_3100_thickeners",
        "uti_6400_h2s",
        "uti_6700_saps_events",
        "ref_nickel_furnace_events",
        "ref_cobalt_furnace_events",
    ]:
        out[c] = 0

    if calendar_df is None or calendar_df.empty:
        return out

    sd_col = _find_col(calendar_df, ["Start date", "Start Date", "Start"])
    ed_col = _find_col(calendar_df, ["End date", "End Date", "End"])
    type_col = _find_col(calendar_df, ["Event type", "Type", "Event"])
    id_col = _find_col(calendar_df, ["Event ID", "Event Id", "EventID", "ID", "Id"])

    if not (sd_col and ed_col and type_col):
        return out

    day_index = {d.date(): i for i, d in enumerate(out["date"])}

    for _, r in calendar_df.iterrows():
        s = _excel_to_datetime(r[sd_col])
        e = _excel_to_datetime(r[ed_col])
        if not (s and e):
            continue
        etype = str(r[type_col]).strip()
        eid = str(r[id_col]).strip().upper() if (id_col and id_col in calendar_df.columns) else ""

        d = s.date()
        end_d = e.date()
        while d <= end_d:
            if d in day_index:
                idx = day_index[d]
                if etype == "TPSD":
                    out.at[idx, "tpsd_active"] = 1
                elif etype == "PAL - 3200 HPALs":
                    out.at[idx, "pal_3200_hpal_events"] += 1
                elif etype == "PAL - 3500 Reactor Swaps":
                    out.at[idx, "pal_3500_reactor_swaps"] += 1
                elif etype == "PAL - 3100 Thickeners":
                    out.at[idx, "pal_3100_thickeners"] += 1
                elif etype == "UTI - 6400 H2S":
                    out.at[idx, "uti_6400_h2s"] += 1
                elif etype == "UTI - 6700 SAPs":
                    out.at[idx, "uti_6700_saps_events"] += 1
                elif etype == "REF - 44/4700 Furnaces":
                    if ("44FR02" in eid) or ("44FR01" in eid):
                        out.at[idx, "ref_nickel_furnace_events"] += 1
                    if "47FR01" in eid:
                        out.at[idx, "ref_cobalt_furnace_events"] += 1
            d = d + timedelta(days=1)

    return out


# ---------- Daily variables (header on second row) ----------
def _parse_daily_variables_from_book(spp_path: str | Path, book: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Read 'Daily variables' with header on the SECOND row (header=1),
    normalize required columns and types. Factors are used AS-IS.
    """
    # find actual sheet name (case-insensitive)
    dv_name = _find_sheet(book, "Daily variables", "Daily Variables", "daily variables")
    if dv_name is None:
        return pd.DataFrame(columns=["date", "standby_autoclaves", "hpal_ext_dt_factor", "hpal_int_dt_factor"])

    # re-read only this sheet with header=1
    dv = pd.read_excel(Path(spp_path), sheet_name=dv_name, engine="openpyxl", header=1)

    # map to canonical names
    rename_map = {}
    for col in dv.columns:
        key = str(col).strip().lower()
        if key == "date":
            rename_map[col] = "date"
        elif ("stand-by" in key or "standby" in key) and "autoclave" in key:
            rename_map[col] = "standby_autoclaves"
        elif "external downtime" in key:
            rename_map[col] = "hpal_ext_dt_factor"
        elif "internal downtime" in key:
            rename_map[col] = "hpal_int_dt_factor"
    dv = dv.rename(columns=rename_map).copy()

    # ensure required cols
    for req in ["date", "standby_autoclaves", "hpal_ext_dt_factor", "hpal_int_dt_factor"]:
        if req not in dv.columns:
            dv[req] = pd.NA

    # types (no scaling)
    dv["date"] = pd.to_datetime(dv["date"], errors="coerce")
    dv = dv.loc[dv["date"].notna()].copy()
    dv["standby_autoclaves"] = pd.to_numeric(dv["standby_autoclaves"], errors="coerce").fillna(0).round().astype(int)
    dv["hpal_ext_dt_factor"] = pd.to_numeric(dv["hpal_ext_dt_factor"], errors="coerce").fillna(0.0)
    dv["hpal_int_dt_factor"] = pd.to_numeric(dv["hpal_int_dt_factor"], errors="coerce").fillna(0.0)

    dv = dv.sort_values("date").reset_index(drop=True)
    return dv


# ---------- public API ----------
def load_inputs(spp_path: str | Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Load all inputs from SPP.xlsx and return:
      constants: dict (from 'Constants')
      daily_variables: DataFrame
        columns at minimum:
          date, standby_autoclaves, hpal_ext_dt_factor, hpal_int_dt_factor,
          tpsd_active, pal_3200_hpal_events, pal_3500_reactor_swaps,
          pal_3100_thickeners, uti_6400_h2s, uti_6700_saps_events,
          ref_nickel_furnace_events, ref_cobalt_furnace_events
    """
    spp_path = Path(spp_path)
    book = pd.read_excel(spp_path, sheet_name=None, engine="openpyxl")

    # --- Constants
    const_name = _find_sheet(book, "Constants", "constants")
    constants = _parse_constants(book.get(const_name))

    # --- Daily variables (header on row 2)
    dv = _parse_daily_variables_from_book(spp_path, book)

    # --- Shutdown Calendar (to enrich daily variables)
    shutdown_name = _find_sheet(book, "Shutdown Calendar", "shutdown_calendar", "Shutdown calendar")
    shutdown_df = book.get(shutdown_name)
    shutdown_daily = _shutdown_calendar_to_daily(shutdown_df, dv["date"])
    daily_variables = dv.merge(shutdown_daily, on="date", how="left")

    # fill any missing marker columns with zeros
    for c in [
        "tpsd_active",
        "pal_3200_hpal_events",
        "pal_3500_reactor_swaps",
        "pal_3100_thickeners",
        "uti_6400_h2s",
        "uti_6700_saps_events",
        "ref_nickel_furnace_events",
        "ref_cobalt_furnace_events",
    ]:
        if c not in daily_variables.columns:
            daily_variables[c] = 0
        daily_variables[c] = pd.to_numeric(daily_variables[c], errors="coerce").fillna(0).astype(int)

    # (We read Mine Plan as well, but no need to return it yet; other modules can be wired later.)

    return constants, daily_variables
