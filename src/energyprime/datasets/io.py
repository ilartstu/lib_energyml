"""Reading CSV / Excel files into a Dataset.

Handles "dirty" files: unknown encoding (utf-8 / cp1251), unknown delimiter,
metadata lines above the header (Open-Meteo puts the header on line 4), a
units row under the header, decimal commas.
"""
from __future__ import annotations

import csv
import io
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.schema import ColumnSpec, StationSpec
from ..utils.validation import infer_freq
from .dataset import Dataset

PREVIEW_ROWS = 40
CSV_EXTS = {".csv", ".txt", ".tsv"}
EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}


# ---------------------------------------------------------------- raw text
def _decode(raw: bytes, encoding: str = "auto") -> str:
    if encoding != "auto":
        return raw.decode(encoding)
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def detect_delimiter(text: str) -> str:
    """Sniff the CSV delimiter from the first non-empty lines; default comma."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    sample = "\n".join(lines)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def guess_header_row(rows: list[list[str]]) -> int:
    """Among text-like rows, pick the one followed by the longest run of data rows.

    Skips metadata blocks (Open-Meteo: lat/lon on row 0, header on row 3).
    """
    def is_data_cell(s: str) -> bool:
        s = s.strip()
        if s == "" or s.lower() in {"nan", "na", "null", "none"}:
            return True
        try:
            float(s.replace(",", "."))
            return True
        except ValueError:
            pass
        return any(ch.isdigit() for ch in s) and any(sep in s for sep in "-/:")

    def is_data_row(row: list[str]) -> bool:
        nonempty = [c for c in row if c.strip() != ""]
        if not nonempty:
            return False
        return sum(is_data_cell(c) for c in nonempty) / len(nonempty) >= 0.6

    def is_blank(row: list[str]) -> bool:
        return all(c.strip() == "" for c in row)

    best_row, best_run = 0, -1
    for i, row in enumerate(rows[:15]):
        nonempty = [c for c in row if c.strip() != ""]
        if len(nonempty) < 2 or is_data_row(row):
            continue
        run = 0
        for following in rows[i + 1:]:
            if is_data_row(following):
                run += 1
            elif is_blank(following):
                continue
            else:
                break
        if run > best_run:
            best_run, best_row = run, i
    return best_row


def _normalize_names(columns) -> list[str]:
    out = []
    for i, c in enumerate(columns):
        s = str(c).strip()
        out.append(s if s else f"col_{i}")
    return out


def raw_preview(path: str | Path, n: int = PREVIEW_ROWS, sheet: str | int | None = None,
                sep: str = "auto", encoding: str = "auto") -> dict:
    """First ``n`` rows as strings plus the suggested header row."""
    path = Path(path)
    if path.suffix.lower() in EXCEL_EXTS:
        df = pd.read_excel(path, sheet_name=sheet or 0, header=None, nrows=n, dtype=object)
        rows = [["" if pd.isna(v) else str(v) for v in r] for r in df.values.tolist()]
    else:
        text = _decode(path.read_bytes(), encoding)
        delim = detect_delimiter(text) if sep == "auto" else sep
        rows = []
        for i, row in enumerate(csv.reader(io.StringIO(text), delimiter=delim)):
            if i >= n:
                break
            rows.append([str(c) for c in row])
    ncols = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    return {"rows": rows, "ncols": ncols, "suggested_header": guess_header_row(rows)}


def read_table(path: str | Path, *, sheet: str | int | None = None, header_row: int | str = "auto",
               skip_after: int = 0, sep: str = "auto", encoding: str = "auto") -> pd.DataFrame:
    """Read a CSV/TXT/TSV or Excel file as-is (no type conversion).

    ``header_row`` is 0-based or ``"auto"``; ``skip_after`` drops rows right
    after the header (e.g. a units row).
    """
    path = Path(path)
    if header_row == "auto":
        header_row = raw_preview(path, sheet=sheet, sep=sep, encoding=encoding)["suggested_header"]
    header_row = int(header_row)
    if path.suffix.lower() in EXCEL_EXTS:
        df = pd.read_excel(path, sheet_name=sheet or 0, header=header_row)
    else:
        text = _decode(path.read_bytes(), encoding)
        delim = detect_delimiter(text) if sep == "auto" else sep
        body = "\n".join(text.splitlines()[header_row:])
        try:
            df = pd.read_csv(io.StringIO(body), sep=delim, low_memory=False)
        except pd.errors.ParserError:
            df = pd.read_csv(io.StringIO(body), sep=delim, engine="python")
    if skip_after > 0:
        df = df.iloc[skip_after:]
    df.columns = _normalize_names(df.columns)
    return df.reset_index(drop=True)


# ------------------------------------------------------------- conversions
def looks_like_time(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.7)


def parse_time(series: pd.Series, format: str | None = None, dayfirst: bool = False) -> pd.DatetimeIndex:
    """Parse timestamps; values that cannot be parsed become NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.DatetimeIndex(series)
    if format is not None:
        return pd.DatetimeIndex(pd.to_datetime(series, format=format, errors="coerce"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series.astype(str), errors="coerce", format="mixed", dayfirst=dayfirst)
    return pd.DatetimeIndex(parsed)


def coerce_numeric(df: pd.DataFrame, exclude=()) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert text columns to numbers ("12,5" -> 12.5).

    Columns that are mostly text stay as they are. Returns the frame and, per
    column, how many non-empty values could not be parsed (they become NaN).
    """
    out = df.copy()
    failed: dict[str, int] = {}
    for col in out.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(out[col]):
            continue
        text = out[col].astype("string").str.strip()
        number = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
        present = text.notna() & (text != "")
        if present.sum() == 0 or number[present].notna().mean() < 0.5:
            continue
        bad = int((present & number.isna()).sum())
        if bad:
            failed[col] = bad
        out[col] = number.astype(float)
    return out, failed


# ------------------------------------------------------------------ loading
def load_table(source: str | Path | pd.DataFrame, *, spec: StationSpec | dict | str | Path | None = None,
               target: str | None = None, time_column: str | None = None,
               time_format: str | None = None, name: str | None = None, **read_kwargs: Any) -> Dataset:
    """Load a file (or a DataFrame) into a :class:`Dataset`.

    Without ``spec`` every numeric column becomes a feature and ``target`` is
    the target. With ``spec`` (StationSpec, dict or YAML path) the roles,
    time format and read options come from it.

    >>> ds = load_table("solar.csv", target="POWER", time_column="Datetime")
    >>> ds = load_table("SPP_2.csv", spec="spp2.yaml")
    """
    spec_obj = StationSpec.load(spec) if spec is not None else None
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
        source_name = name or "dataframe"
    else:
        options = dict(vars(spec_obj.read)) if spec_obj else {}
        options.pop("rename", None)
        options.update(read_kwargs)
        raw = read_table(source, **options)
        source_name = str(source)
    if spec_obj and spec_obj.read.rename:
        raw = raw.rename(columns=spec_obj.read.rename)

    time_col = time_column or (spec_obj.time.column if spec_obj else None)
    if time_col is None:
        time_col = next((c for c in raw.columns if looks_like_time(raw[c])), None)
        if time_col is None:
            raise ValueError("Could not find a time column; pass time_column=...")
    if time_col not in raw.columns:
        raise KeyError(f"Time column {time_col!r} not found; columns: {list(raw.columns)}")

    fmt = time_format or (spec_obj.time.format if spec_obj else None)
    dayfirst = spec_obj.time.dayfirst if spec_obj else False
    index = parse_time(raw[time_col], fmt, dayfirst)
    bad_time = np.asarray(index.isna())
    df = raw.loc[~bad_time].drop(columns=[time_col])
    df.index = index[~bad_time]
    df.index.name = time_col
    df, coerced = coerce_numeric(df)

    if spec_obj and spec_obj.time.timezone:
        df.index = df.index.tz_localize(spec_obj.time.timezone) if df.index.tz is None \
            else df.index.tz_convert(spec_obj.time.timezone)

    was_sorted = bool(df.index.is_monotonic_increasing)
    if not was_sorted:
        df = df.sort_index(kind="stable")

    if spec_obj is None:
        spec_obj = StationSpec.infer(df, target=target, name=name or Path(source_name).stem)
        spec_obj.time.column = time_col
        spec_obj.time.format = fmt
    else:
        spec_obj = spec_obj.copy()
        spec_obj.time.column = time_col
        if target is not None:
            for col_name, col in spec_obj.columns.items():
                if col.role == "target" and col_name != target:
                    col.role = "feature"
            if target in spec_obj.columns:
                spec_obj.columns[target].role = "target"
            else:
                spec_obj.columns[target] = ColumnSpec(target, role="target")

    missing_columns = [c for c in spec_obj.columns if c not in df.columns]
    if missing_columns:
        warnings.warn(f"Columns from the spec not found in the data: {missing_columns}", stacklevel=2)
    if spec_obj.target is not None and spec_obj.target not in df.columns:
        raise KeyError(f"Target column {spec_obj.target!r} not found in the data")

    freq = infer_freq(df.index)
    info = {
        "source": source_name,
        "rows": int(len(df)),
        "unparsed_time_rows": int(bad_time.sum()),
        "values_not_numeric": coerced,
        "was_sorted": was_sorted,
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "missing_columns": missing_columns,
        "ignored_columns": [c for c in df.columns if spec_obj.column(c) is None
                            or spec_obj.column(c).role == "ignore"],
        "inferred_freq": str(freq) if freq is not None else None,
    }
    return Dataset(df, spec_obj, info)


def load_station(source: str | Path | pd.DataFrame, spec: StationSpec | dict | str | Path,
                 **kwargs: Any) -> Dataset:
    """Load station data with its YAML spec (roles, time format, cleaning rules)."""
    return load_table(source, spec=spec, **kwargs)
