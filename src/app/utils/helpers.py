import csv
import json
import os
import threading
import time
from io import BytesIO

import chardet
import pandas as pd
from fastapi import HTTPException

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    UPLOAD_TTL_SECONDS = int(os.environ.get("UPLOAD_TTL_SECONDS", 60 * 60 * 24))
except Exception:
    UPLOAD_TTL_SECONDS = 60 * 60 * 24

# Control whether uploads are persisted to disk. Set to '0' or 'false' to keep uploads
# in-memory (useful for avoiding accumulation of many temporary files during tests).
try:
    UPLOAD_PERSIST = str(os.environ.get("UPLOAD_PERSIST", "1")).lower() not in (
        "0",
        "false",
        "no",
    )
except Exception:
    UPLOAD_PERSIST = True

# In-memory temporary upload store when persistence is disabled.
_TEMP_UPLOADS: dict[str, tuple[float, bytes]] = {}

# Track whether the background cleanup worker is already running to avoid spawning duplicates.
_cleanup_worker_started = False
_cleanup_worker_lock = threading.Lock()


def save_temp_upload(file_id: str, b: bytes):
    """Save an uploaded file in memory (used when UPLOAD_PERSIST is False)."""
    try:
        _TEMP_UPLOADS[file_id] = (time.time(), b)
    except Exception:
        pass


def _upload_persist_enabled() -> bool:
    """Determine if uploads should persist to disk (honours runtime env overrides)."""
    try:
        override = os.environ.get("UPLOAD_PERSIST")
        if override is not None:
            return str(override).lower() not in ("0", "false", "no")
    except Exception:
        pass
    return UPLOAD_PERSIST


def get_temp_upload(file_id: str) -> bytes | None:
    item = _TEMP_UPLOADS.get(file_id)
    if not item:
        return None
    return item[1]


def cleanup_uploads(now=None, ttl_override=None):
    """Remove expired uploads on disk and in-memory based on TTL (in seconds)."""
    now = now or time.time()
    removed = []
    # determine TTL: override takes precedence, then env var, then default
    try:
        if ttl_override is not None:
            ttl = int(ttl_override)
        else:
            ttl = int(os.environ.get("UPLOAD_TTL_SECONDS", UPLOAD_TTL_SECONDS))
    except Exception:
        ttl = UPLOAD_TTL_SECONDS
    # cleanup on-disk uploads
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            mtime = os.path.getmtime(path)
            age = now - mtime
            if age < 0:
                age = 0
            if age >= ttl:
                os.remove(path)
                removed.append(name)
        except Exception:
            continue
    # cleanup in-memory uploads
    try:
        to_del = []
        for fid, (ts, _b) in list(_TEMP_UPLOADS.items()):
            age = now - ts
            if age < 0:
                age = 0
            if age >= ttl:
                to_del.append(fid)
        for fid in to_del:
            _TEMP_UPLOADS.pop(fid, None)
            removed.append(fid)
    except Exception:
        pass
    return removed


def _start_cleanup_worker():
    global _cleanup_worker_started
    if _cleanup_worker_started:
        return

    with _cleanup_worker_lock:
        if _cleanup_worker_started:
            return

        def _cleanup_worker():
            while True:
                try:
                    cleanup_uploads()
                except Exception:
                    pass
                time.sleep(60 * 60)

        threading.Thread(target=_cleanup_worker, daemon=True).start()
        _cleanup_worker_started = True


API_KEY = os.environ.get("ASTPARSER_API_KEY")


def _require_api_key(headers: dict):
    if not API_KEY:
        return True
    for k, v in headers.items():
        if k.lower() == "x-api-key" and v == API_KEY:
            return True
    return False


def _detect_csv_separator(sample_text: str) -> str:
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text)
        return dialect.delimiter
    except Exception:
        return ","


def _detect_encoding(b: bytes) -> str:
    try:
        guess = chardet.detect(b)
        if guess and guess.get("encoding"):
            return guess["encoding"]
    except Exception:
        pass
    return "utf-8"


def _loads_join(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return str(s).strip()


def _find_csv_data_start(text: str, delimiters=None, min_fields=3):
    if delimiters is None:
        delimiters = [",", ";", "\t", "|"]
    lines = text.splitlines()
    best = (0, ",", 0)
    for delim in delimiters:
        counts = [len(line.split(delim)) for line in lines]
        for i, c in enumerate(counts):
            if c >= min_fields:
                window = counts[i : i + 10]
                valid = sum(1 for x in window if x >= min_fields)
                if valid >= 1 and c > best[2]:
                    best = (i, delim, c)
                break
    return best[0], best[1]


def read_preview_from_bytes(
    b: bytes,
    filename: str,
    nrows: int = 20,
    has_header: bool = True,
    delimiter: str | None = None,
):
    ext = filename.lower().split(".")[-1]
    header = 0 if has_header else None
    if ext in ("xls", "xlsx"):
        df = pd.read_excel(BytesIO(b), engine="openpyxl", nrows=nrows, header=header)
    elif ext == "csv":
        enc = _detect_encoding(b)
        text = b.decode(enc, errors="replace")
        start_idx, detected_delim = _find_csv_data_start(text)
        lines = text.splitlines()
        data_text = "\n".join(lines[start_idx:])
        first_lines = "\n".join(lines[start_idx : start_idx + 10])

        df = None
        if delimiter:
            try:
                df = pd.read_csv(
                    BytesIO(data_text.encode(enc)),
                    sep=delimiter,
                    engine="python",
                    nrows=nrows,
                    header=header,
                    encoding=enc,
                )
            except Exception:
                df = None

        if df is None:
            try:
                sep_sniff = delimiter or detected_delim or _detect_csv_separator(first_lines)
                df = pd.read_csv(
                    BytesIO(data_text.encode(enc)),
                    sep=sep_sniff,
                    engine="python",
                    nrows=nrows,
                    header=header,
                    encoding=enc,
                )
            except Exception:
                df = None

        if df is None:
            try:
                df = pd.read_csv(
                    BytesIO(data_text.encode(enc)),
                    sep=None,
                    engine="python",
                    nrows=nrows,
                    header=header,
                    encoding=enc,
                )
            except Exception:
                df = None

        if df is None:
            for sep_try in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(
                        BytesIO(data_text.encode(enc)),
                        sep=sep_try,
                        engine="python",
                        nrows=nrows,
                        header=header,
                        encoding=enc,
                    )
                    if df is not None:
                        break
                except Exception:
                    df = None

        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse CSV with common delimiters")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    df = df.fillna("")
    cols = [str(c) for c in df.columns]
    preview = df.head(nrows).astype(str).to_dict(orient="records")
    return cols, preview


def read_full_dataframe_from_path(path: str, filename: str, has_header: bool = True, delimiter: str | None = None):
    with open(path, "rb") as f:
        b = f.read()
    ext = filename.lower().split(".")[-1]
    header = 0 if has_header else None
    if ext in ("xls", "xlsx"):
        df = pd.read_excel(BytesIO(b), engine="openpyxl", header=header)
    elif ext == "csv":
        enc = _detect_encoding(b)
        text = b.decode(enc, errors="replace")
        start_idx, detected_delim = _find_csv_data_start(text)
        lines = text.splitlines()
        data_text = "\n".join(lines[start_idx:])
        first_lines = "\n".join(lines[start_idx : start_idx + 10])

        df = None
        if delimiter:
            try:
                df = pd.read_csv(
                    BytesIO(data_text.encode(enc)),
                    sep=delimiter,
                    engine="python",
                    header=header,
                    encoding=enc,
                )
            except Exception:
                df = None

        if df is None:
            try:
                sep_sniff = delimiter or detected_delim or _detect_csv_separator(first_lines)
                df = pd.read_csv(
                    BytesIO(data_text.encode(enc)),
                    sep=sep_sniff,
                    engine="python",
                    header=header,
                    encoding=enc,
                )
            except Exception:
                df = None

        if df is None:
            try:
                df = pd.read_csv(
                    BytesIO(data_text.encode(enc)),
                    sep=None,
                    engine="python",
                    header=header,
                    encoding=enc,
                )
            except Exception:
                df = None

        if df is None:
            for sep_try in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(
                        BytesIO(data_text.encode(enc)),
                        sep=sep_try,
                        engine="python",
                        header=header,
                        encoding=enc,
                    )
                    if df is not None:
                        break
                except Exception:
                    df = None

        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse CSV with common delimiters")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file extension")
    return df


def dataframe_to_csv_stream(df: pd.DataFrame):
    def gen():
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        yield buf.read()

    return gen


def _resolve_columns(df: pd.DataFrame, requested: list | None):
    if not requested:
        return list(df.columns)
    resolved = []
    for item in requested:
        if isinstance(item, int):
            try:
                resolved.append(df.columns[item])
                continue
            except Exception:
                pass
        try:
            idx = int(str(item))
            resolved.append(df.columns[idx])
            continue
        except Exception:
            pass
        if str(item) in df.columns:
            resolved.append(str(item))
            continue
        for c in df.columns:
            if str(c) == str(item):
                resolved.append(c)
                break
    seen = set()
    out = []
    for c in resolved:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _resolve_row_indices(df: pd.DataFrame, requested: list | None):
    if not requested:
        return list(range(len(df)))
    out = []
    for item in requested:
        try:
            if isinstance(item, int):
                out.append(int(item))
                continue
        except Exception:
            pass
        try:
            val = int(item)
            out.append(val)
            continue
        except Exception:
            pass
        try:
            s = str(item)
            if "-" in s:
                a, b = s.split("-", 1)
                a = int(a)
                b = int(b)
                out.extend(list(range(a, b + 1)))
            else:
                out.append(int(s))
        except Exception:
            continue
    valid = [i for i in out if 0 <= i < len(df)]
    return valid


def _prepare_df_for_selection(file_id: str):
    """Return a pandas DataFrame for a stored upload.

    If `UPLOAD_PERSIST` is True the file is read from disk under `uploads/`.
    If persistence is disabled, the bytes are read from an in-memory store.
    """
    path = os.path.join(UPLOAD_DIR, file_id)
    if _upload_persist_enabled():
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"file not found: {file_id}")
        return read_full_dataframe_from_path(path, file_id)
    # persistence disabled -> look up in-memory
    item = _TEMP_UPLOADS.get(file_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"file not found (in-memory): {file_id}")
    _, b = item
    # Use the same parsing routine but from bytes; implement a small helper here
    ext = file_id.lower().split(".")[-1]
    header = 0
    if ext in ("xls", "xlsx"):
        try:
            return pd.read_excel(BytesIO(b), engine="openpyxl", header=header)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read excel: {e}") from e
    elif ext == "csv":
        try:
            enc = _detect_encoding(b)
            text = b.decode(enc, errors="replace")
            start_idx, detected_delim = _find_csv_data_start(text)
            lines = text.splitlines()
            data_text = "\n".join(lines[start_idx:])
            sep_sniff = detected_delim or _detect_csv_separator("\n".join(lines[start_idx : start_idx + 10]))
            return pd.read_csv(
                BytesIO(data_text.encode(enc)),
                sep=sep_sniff,
                engine="python",
                header=header,
                encoding=enc,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV from bytes: {e}") from e
    else:
        raise HTTPException(status_code=400, detail="Unsupported file extension")


def _apply_selection_for_compare(df: pd.DataFrame, sel_cols, sel_rows, ex_cols, ex_rows, mode: str):
    df2 = df.copy()
    if mode in ("columns", "both"):
        cols = _resolve_columns(df2, sel_cols)
        df2 = df2.loc[:, cols]
        if ex_cols:
            ex = _resolve_columns(df2, ex_cols)
            df2 = df2.drop(columns=ex)
    if mode in ("rows", "both"):
        rows_idx = _resolve_row_indices(df, sel_rows)
        # use positional indexing for rows
        df2 = df.iloc[rows_idx]
        if ex_rows:
            ex_pos = set(_resolve_row_indices(df, ex_rows))
            keep_positions = [i for i in rows_idx if i not in ex_pos]
            df2 = df.iloc[keep_positions][df2.columns]
    return df2


def _compare_dfs(
    df_a,
    df_b,
    mode: str,
    a_sel_cols,
    a_sel_rows,
    a_ex_cols,
    a_ex_rows,
    b_sel_cols,
    b_sel_rows,
    b_ex_cols,
    b_ex_rows,
    a_join_on=None,
    b_join_on=None,
):
    df_a2 = _apply_selection_for_compare(df_a, a_sel_cols, a_sel_rows, a_ex_cols, a_ex_rows, mode)
    df_b2 = _apply_selection_for_compare(df_b, b_sel_cols, b_sel_rows, b_ex_cols, b_ex_rows, mode)

    if a_join_on and b_join_on:

        def _resolve_join_col(df, col):
            if col is None:
                return None
            try:
                ci = int(str(col))
                return df.columns[ci]
            except Exception:
                return col

        a_key = _resolve_join_col(df_a2, a_join_on)
        b_key = _resolve_join_col(df_b2, b_join_on)
        if a_key not in df_a2.columns or b_key not in df_b2.columns:
            raise HTTPException(status_code=400, detail="join-on column not found")

        df_a2["_ASTPARSER__join_key"] = df_a2[a_key].astype(str)
        df_b2["_ASTPARSER__join_key"] = df_b2[b_key].astype(str)
        merged = pd.merge(
            df_a2.reset_index(drop=True),
            df_b2.reset_index(drop=True),
            on="_ASTPARSER__join_key",
            how="inner",
            suffixes=("_A", "_B"),
        )
        if merged.shape[0] == 0:
            df_a2 = df_a2.head(0)
            df_b2 = df_b2.head(0)
        else:
            a_cols_final = [c for c in merged.columns if c.endswith("_A") and c != "_ASTPARSER__join_key"]
            b_cols_final = [c for c in merged.columns if c.endswith("_B") and c != "_ASTPARSER__join_key"]
            a_cols_readable = [c[:-2] for c in a_cols_final]
            b_cols_readable = [c[:-2] for c in b_cols_final]
            df_a2 = merged[a_cols_final]
            df_a2.columns = a_cols_readable
            df_b2 = merged[b_cols_final]
            df_b2.columns = b_cols_readable

    min_rows = min(len(df_a2.index), len(df_b2.index))
    min_cols = min(len(df_a2.columns), len(df_b2.columns))

    result = []
    cols = []
    for cidx in range(min_cols):
        ca = str(df_a2.columns[cidx])
        cb = str(df_b2.columns[cidx])
        cols.append((ca, cb))

    for ridx in range(min_rows):
        ra = df_a2.iloc[ridx]
        rb = df_b2.iloc[ridx]
        row = {}
        for cidx in range(min_cols):
            ca, cb = cols[cidx]
            aval = ra.iloc[cidx]
            bval = rb.iloc[cidx]
            aval_py = aval if isinstance(aval, (str, float, int, bool)) else str(aval)
            bval_py = bval if isinstance(bval, (str, float, int, bool)) else str(bval)
            match = str(aval_py) == str(bval_py)
            row[f"A::{ca}"] = aval_py
            row[f"B::{cb}"] = bval_py
            row[f"MATCH::{cidx}"] = match
            if not match:
                row[f"DIFF::{cidx}"] = f"{aval} -> {bval}"
        result.append(row)

    columns_out = [f"A::{a[0]}" for a in cols] + [f"B::{a[1]}" for a in cols] + [f"MATCH::{i}" for i in range(min_cols)]
    return {"rows": result, "columns": columns_out}


# start background cleanup worker
_start_cleanup_worker()
