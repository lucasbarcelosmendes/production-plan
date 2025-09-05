# models/reagent_forecast.py
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

EXCEL_EPOCH = datetime(1899, 12, 30)

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _parse_month_label(label: Any) -> Optional[tuple[int, int]]:
    """
    Accept month headers as:
      - pandas/py datetimes: Timestamp('2025-01-01') -> (2025, 1)
      - Excel serials: 45292 -> (YYYY, MM)
      - Strings: 'Jan-25', 'Jan-2025', 'January 2025', '2025-01', etc.
    """
    if label is None:
        return None

    # direct datetime-like
    if isinstance(label, pd.Timestamp):
        return label.year, label.month
    if isinstance(label, datetime):
        return label.year, label.month

    # excel serial in header
    if isinstance(label, (int, float)) and not pd.isna(label):
        try:
            dt = EXCEL_EPOCH + timedelta(days=float(label))
            return dt.year, dt.month
        except Exception:
            pass

    s = str(label).strip()
    if not s:
        return None

    # 2025-01 or 2025/1, etc.
    m = re.fullmatch(r"(\d{4})[^\d]?(\d{1,2})", s)
    if m:
        y = int(m.group(1)); mm = int(m.group(2))
        if 1 <= mm <= 12:
            return y, mm

    # normalize separators
    s_norm = re.sub(r"[ _/.,]+", "-", s)

    # Mon-YY / Mon-YYYY / Month-YYYY
    parts = s_norm.split("-")
    if len(parts) == 2:
        p0, p1 = parts[0].strip(), parts[1].strip()
        mm = _MONTHS.get(p0.lower()) if not p0.isdigit() else int(p0)
        if mm and 1 <= mm <= 12 and p1.isdigit():
            if len(p1) == 2:
                y = 2000 + int(p1)
            elif len(p1) == 4:
                y = int(p1)
            else:
                return None
            return y, mm

    # Month YYYY
    m2 = re.fullmatch(r"([A-Za-z]+)[- ]+(\d{4})", s_norm)
    if m2:
        mm = _MONTHS.get(m2.group(1).lower())
        y = int(m2.group(2))
        if mm:
            return y, mm

    return None

def compute_reagent_daily_calendar(reagent_table: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a monthly reagent forecast into a daily calendar.

    Input example:
        Reagent | Jan-25 | Feb-25 | ...
        NaSH    |   10   |   12   | ...
        Urea    |    0   |    0   | ...

    Output:
        Date (YYYY-MM-DD) | NaSH | Urea | ...
    """
    if reagent_table is None or reagent_table.empty:
        return pd.DataFrame(columns=["Date"])

    reagent_col = _find_col(reagent_table, ["Reagent"])
    if reagent_col is None:
        return pd.DataFrame(columns=["Date"])

    # detect month columns (headers may be strings, timestamps, or serials)
    month_cols: List[str] = []
    parsed_map: Dict[str, tuple[int, int]] = {}
    for c in reagent_table.columns:
        if c == reagent_col:
            continue
        ym = _parse_month_label(c)
        if ym:
            month_cols.append(c)
            parsed_map[str(c)] = ym

    if not month_cols:
        # no recognizable month headers
        return pd.DataFrame(columns=["Date"])

    daily_map: Dict[str, Dict[str, Any]] = {}

    for _, row in reagent_table.iterrows():
        reagent_name = str(row[reagent_col]).strip() if pd.notna(row[reagent_col]) else ""
        if not reagent_name:
            continue

        for mcol in month_cols:
            year, month = parsed_map[str(mcol)]
            try:
                per_day_val = float(row[mcol]) if pd.notna(row[mcol]) else 0.0
            except Exception:
                per_day_val = 0.0

            _, ndays = calendar.monthrange(year, month)
            for day in range(1, ndays + 1):
                d = date(year, month, day).isoformat()
                rec = daily_map.get(d)
                if rec is None:
                    rec = {"Date": d}
                    daily_map[d] = rec
                rec[reagent_name] = per_day_val

    out = pd.DataFrame(list(daily_map.values()))
    if out.empty:
        return pd.DataFrame(columns=["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    out["Date"] = out["Date"].astype(str)
    return out
