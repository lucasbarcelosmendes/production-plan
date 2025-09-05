# models/mine_plan_daily.py
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

EXCEL_EPOCH = datetime(1899, 12, 30)

# -------- helpers --------

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

def _parse_month_label(label: Any) -> Optional[Tuple[int, int]]:
    """Accept headers as Timestamps, datetimes, Excel serials, or strings like 'Jan-25', '2025-01', 'January 2025'."""
    if label is None:
        return None
    if isinstance(label, pd.Timestamp):
        return label.year, label.month
    if isinstance(label, datetime):
        return label.year, label.month
    if isinstance(label, (int, float)) and not pd.isna(label):
        # Try Excel serial
        try:
            dt = EXCEL_EPOCH + timedelta(days=float(label))
            return dt.year, dt.month
        except Exception:
            pass
    s = str(label).strip()
    if not s:
        return None
    # 2025-01 or 2025/1 or 2025.1
    m = re.fullmatch(r"(\d{4})[^\d]?(\d{1,2})", s)
    if m:
        y, mm = int(m.group(1)), int(m.group(2))
        return (y, mm) if 1 <= mm <= 12 else None
    s_norm = re.sub(r"[ _/.,]+", "-", s)
    # Jan-25 or 1-2025
    parts = s_norm.split("-")
    if len(parts) == 2:
        p0, p1 = parts[0].strip(), parts[1].strip()
        if p0.isdigit() and p1.isdigit():
            mm, y = int(p0), int(p1 if len(p1) == 4 else 2000 + int(p1))
            if 1 <= mm <= 12:
                return y, mm
        else:
            mm = _MONTHS.get(p0.lower())
            if mm and p1.isdigit():
                y = int(p1 if len(p1) == 4 else 2000 + int(p1))
                return (y, mm)
    # Month YYYY
    m2 = re.fullmatch(r"([A-Za-z]+)[- ]+(\d{4})", s_norm)
    if m2:
        mm = _MONTHS.get(m2.group(1).lower()); y = int(m2.group(2))
        if mm:
            return y, mm
    return None

def _to_float_if_numeric(x: Any) -> Any:
    """Try to coerce to float; if it fails, return original (keeps text columns)."""
    try:
        # Treat empty strings as NaN -> np.nan (kept as NaN)
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return x

def _coerce_series_mixed(s: pd.Series) -> pd.Series:
    """Coerce series to numeric where possible, keeping non-numeric values as-is."""
    # Use to_numeric with errors='ignore' to keep strings; then clean empty strings
    out = pd.to_numeric(s, errors="ignore")
    if out.dtype == object:
        return out.replace({"": np.nan})
    return out

def _normalize_date_value(v: Any) -> Optional[str]:
    """Convert potential date values (string, timestamp, excel serial) to ISO yyyy-mm-dd."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.date().isoformat()
    # Excel serial?
    if isinstance(v, (int, float)):
        try:
            dt = EXCEL_EPOCH + timedelta(days=float(v))
            return dt.date().isoformat()
        except Exception:
            pass
    # Parse date-like strings
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if pd.notna(ts):
            return ts.date().isoformat()
    except Exception:
        pass
    # Fallback: return as string; caller may drop invalid
    s = str(v).strip()
    try:
        ts = pd.to_datetime(s, errors="coerce", dayfirst=False)
        if pd.notna(ts):
            return ts.date().isoformat()
    except Exception:
        pass
    return None

def _make_unique(names: List[str]) -> List[str]:
    """Ensure column/metric names are unique by appending (.1), (.2)... to duplicates."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        base = str(n)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}({seen[base]})")
    return out

# -------- main --------

def compute_mine_plan_daily_calendar(mine_plan_sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily mine plan, passing through *all* parameters from the Mine Plan.
    Output: one row per day with a 'Date' column plus one column per parameter.

    Input accepted:
      1) Daily table with a 'Date' column (any case) and arbitrary parameter columns.
      2) Monthly matrix: first column is parameter/metric name; subsequent columns are months.
         Values are repeated across all days in their month.

    Backward-compatible columns (if present) remain as-is:
      - 'Ni in PAL Feed (%)', 'Co in PAL Feed (%)', 'Acid ratio (kg/t)'
    """
    if mine_plan_sheet is None or mine_plan_sheet.empty:
        return pd.DataFrame(columns=["Date"])

    # --- Case 1: already daily ------------------------------------------------
    date_col = _find_col(mine_plan_sheet, ["Date"])
    if date_col is not None:
        df = mine_plan_sheet.copy()

        # Normalize Date
        df["Date"] = df[date_col].apply(_normalize_date_value)
        df = df[df["Date"].notna()]  # drop rows without a parseable date

        # Keep ALL other columns, coercing numerics where it makes sense
        other_cols = [c for c in df.columns if c != date_col and c != "Date"]
        for c in other_cols:
            df[c] = _coerce_series_mixed(df[c])

        # Reorder with Date first
        cols = ["Date"] + [c for c in other_cols]
        # If there are accidental duplicate names, make them unique (excluding Date)
        unique_tail = _make_unique(cols[1:])
        df.columns = ["Date"] + unique_tail
        return df[["Date"] + unique_tail].sort_values("Date").reset_index(drop=True)

    # --- Case 2: monthly matrix ----------------------------------------------
    # Identify metric/parameter name column (first non-month column). Prefer common labels.
    metric_col = _find_col(mine_plan_sheet, ["Mining", "Metric", "Parameter", "Key", "Name", "Item"])
    if metric_col is None:
        metric_col = mine_plan_sheet.columns[0]

    dfm = mine_plan_sheet.copy()
    dfm[metric_col] = dfm[metric_col].astype(str).str.strip()

    # Identify month columns
    month_cols: List[Any] = []
    parsed: Dict[Any, Tuple[int, int]] = {}
    for c in dfm.columns:
        if c == metric_col:
            continue
        ym = _parse_month_label(c)
        if ym:
            month_cols.append(c)
            parsed[c] = ym

    if not month_cols:
        # Nothing to expand
        return pd.DataFrame(columns=["Date"])

    # Ensure unique parameter names
    param_names = _make_unique(dfm[metric_col].tolist())
    dfm = dfm.copy()
    dfm[metric_col] = param_names

    # Build daily records: for each month column, replicate all parameter values across each day.
    daily_records: List[Dict[str, Any]] = []
    for c in month_cols:
        year, month = parsed[c]
        _, ndays = calendar.monthrange(year, month)

        # Grab a dict of parameter -> value for this month
        month_vals = {
            row[metric_col]: _to_float_if_numeric(row.get(c, np.nan))
            for _, row in dfm.iterrows()
        }

        # Create one record per day
        for d in range(1, ndays + 1):
            rec: Dict[str, Any] = {"Date": date(year, month, d).isoformat()}
            rec.update(month_vals)
            daily_records.append(rec)

    out = pd.DataFrame(daily_records)

    # Coerce columns to numeric where possible (exclude Date)
    for c in out.columns:
        if c == "Date":
            continue
        out[c] = _coerce_series_mixed(out[c])

    # Sort & reset
    out = out.sort_values("Date").reset_index(drop=True)

    return out
