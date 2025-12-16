import io
import json
import os
import zipfile
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils.helpers import (
    dataframe_to_csv_stream,
    read_preview_from_bytes,
    save_temp_upload,
)

router = APIRouter()

# Module-level singleton to avoid calling File() in function default arguments
UPLOAD_FILE = File(...)


@router.post(
    "/api/upload-preview",
    tags=["Parsing"],
    summary="Upload and preview a CSV or Excel file",
    description="Upload endpoint that returns a preview of the uploaded file with column names and sample rows.",
    responses={
        200: {
            "description": "Preview with generated file ID",
            "content": {
                "application/json": {
                    "example": {
                        "file_id": "abcd1234_sample.csv",
                        "filename": "sample.csv",
                        "columns": ["col1", "col2", "col3"],
                        "preview": [
                            {"col1": "val1", "col2": "val2", "col3": "val3"},
                            {"col1": "val4", "col2": "val5", "col3": "val6"},
                        ],
                    }
                }
            },
        }
    },
)
async def upload_preview(
    file: UploadFile = UPLOAD_FILE,
    has_header: bool = Form(True, description="Whether the first row contains column headers (e.g., true)"),
    delimiter: str | None = Form(
        None,
        description="CSV delimiter character (auto-detected if not provided) (e.g., ,)",
    ),
    persist: bool | None = Form(
        None,
        description="Whether to persist upload to disk (overrides environment setting) (e.g., false)",
    ),
):
    contents = await file.read()
    file_id = uuid4().hex
    safe_name = f"{file_id}_{file.filename}"
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads", safe_name)
    # determine persistence: dynamic from environment, override by request
    env_persist = os.environ.get("UPLOAD_PERSIST", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    use_persist = env_persist if persist is None else bool(persist)
    if use_persist:
        # persist upload to disk
        with open(path, "wb") as f:
            f.write(contents)
    else:
        # store in-memory to avoid file system accumulation during tests
        try:
            save_temp_upload(safe_name, contents)
        except Exception:
            # fallback to writing to disk if in-memory store fails
            with open(path, "wb") as f:
                f.write(contents)

    try:
        cols, preview = read_preview_from_bytes(
            contents,
            file.filename,
            nrows=20,
            has_header=has_header,
            delimiter=delimiter,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse preview: {e}") from e

    return JSONResponse(
        {
            "file_id": safe_name,
            "filename": file.filename,
            "columns": cols,
            "preview": preview,
        }
    )


@router.post(
    "/api/cleanup-uploads",
    tags=["Parsing"],
    summary="Cleanup expired uploads",
    description="Delete expired on-disk and in-memory uploads based on TTL override.",
    responses={
        200: {
            "description": "List of removed file IDs",
            "content": {"application/json": {"example": {"removed": ["abcd1234_sample.csv"]}}},
        }
    },
)
async def api_cleanup_uploads(
    admin_key: str | None = Form(
        None,
        description="Admin key for authorization (must match ASTPARSER_ADMIN_KEY env var) (e.g., supersecretkey)",
    ),
    ttl: int | None = Form(
        None,
        description="Time-to-live override in seconds (files older than this will be removed) (e.g., 3600)",
    ),
):
    """Cleanup old uploaded files. If `ASTPARSER_ADMIN_KEY` is set in the
    environment, the same value must be provided in the `admin_key` form field
    (tests call without a key when the env var is not set).
    """
    from ..utils.helpers import cleanup_uploads

    admin_env = os.environ.get("ASTPARSER_ADMIN_KEY")
    if admin_env and admin_key != admin_env:
        raise HTTPException(status_code=403, detail="forbidden")
    removed = cleanup_uploads(ttl_override=ttl)
    return JSONResponse({"removed": removed})


@router.post(
    "/api/parse",
    tags=["Parsing"],
    summary="Parse and select data from uploaded file",
    description="Select rows and/or columns from the uploaded file and return JSON records.",
    responses={
        200: {
            "description": "Parsed data with selected columns and rows",
            "content": {
                "application/json": {
                    "example": {
                        "columns": ["col1", "col3"],
                        "rows": [{"col1": "1", "col3": "3"}],
                    }
                }
            },
        }
    },
)
async def parse(
    file_id: str = Form(
        ...,
        description="File identifier returned from /api/upload-preview (e.g., 9f2d1c3e_sample.csv)",
    ),
    mode: str = Form(..., description="Selection mode: 'columns', 'rows', or 'both' (e.g., columns)"),
    selected_columns: str | None = Form(
        None,
        description='JSON array of column names/indices to include (e.g., ["col1", "col3"])',
    ),
    selected_rows: str | None = Form(
        None,
        description="JSON array of row indices to include (e.g., [0, 2, 4])",
    ),
    exclude_columns: str | None = Form(
        None,
        description='JSON array of column names/indices to exclude (e.g., ["col2"])',
    ),
    exclude_rows: str | None = Form(
        None,
        description="JSON array of row indices to exclude (e.g., [1, 3])",
    ),
):
    # Use helper to prepare df (handles in-memory or persisted uploads)
    from ..utils.helpers import _prepare_df_for_selection

    def _loads(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    sel_cols = None
    sel_rows = None
    ex_cols = None
    ex_rows = None
    try:
        import json

        sel_cols = _loads(selected_columns)
        sel_rows = _loads(selected_rows)
        ex_cols = _loads(exclude_columns)
        ex_rows = _loads(exclude_rows)
    except Exception:
        pass

    df = _prepare_df_for_selection(file_id)

    # reuse the selection logic from helpers by importing internals (quick path)
    from ..utils.helpers import _resolve_columns, _resolve_row_indices

    try:
        if mode == "columns":
            cols = _resolve_columns(df, sel_cols)
            df2 = df.loc[:, cols]
            if ex_cols:
                ex = _resolve_columns(df2, ex_cols)
                df2 = df2.drop(columns=ex)
        elif mode == "rows":
            rows_idx = _resolve_row_indices(df, sel_rows)
            df2 = df.iloc[rows_idx]
            if ex_rows:
                ex_pos = set(_resolve_row_indices(df, ex_rows))
                keep_positions = [i for i in rows_idx if i not in ex_pos]
                df2 = df.iloc[keep_positions]
        elif mode == "both":
            cols = _resolve_columns(df, sel_cols)
            rows_idx = _resolve_row_indices(df, sel_rows)
            df2 = df.iloc[rows_idx][cols]
            if ex_cols:
                ex = _resolve_columns(df2, ex_cols)
                df2 = df2.drop(columns=ex)
            if ex_rows:
                ex_pos = set(_resolve_row_indices(df, ex_rows))
                keep_positions = [i for i in rows_idx if i not in ex_pos]
                df2 = df.iloc[keep_positions][cols]
        else:
            raise HTTPException(status_code=400, detail="unknown mode")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Selection error ({type(e).__name__}): {e}") from e

    df2 = df2.fillna("")
    cols_out = [str(c) for c in df2.columns]
    rows_out = df2.astype(str).to_dict(orient="records")
    return JSONResponse({"columns": cols_out, "rows": rows_out})


@router.post(
    "/api/parse-download",
    tags=["Parsing"],
    summary="Parse, select and download CSV",
    description="Select rows and/or columns from the uploaded file and download as CSV.",
    responses={200: {"description": "CSV file download response"}},
)
async def parse_download(
    file_id: str = Form(
        ...,
        description="File identifier returned from /api/upload-preview (e.g., 9f2d1c3e_sample.csv)",
    ),
    mode: str = Form(..., description="Selection mode: 'columns', 'rows', or 'both' (e.g., rows)"),
    selected_columns: str | None = Form(
        None,
        description='JSON array of column names/indices to include (e.g., ["col1", "col3"])',
    ),
    selected_rows: str | None = Form(
        None,
        description="JSON array of row indices to include (e.g., [0, 2, 4])",
    ),
    exclude_columns: str | None = Form(
        None,
        description='JSON array of column names/indices to exclude (e.g., ["col2"])',
    ),
    exclude_rows: str | None = Form(
        None,
        description="JSON array of row indices to exclude (e.g., [1, 3])",
    ),
):
    # Use helper to prepare df (handles in-memory or persisted uploads)
    from ..utils.helpers import _prepare_df_for_selection

    def _loads(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    sel_cols = _loads(selected_columns)
    sel_rows = _loads(selected_rows)
    ex_cols = _loads(exclude_columns)
    ex_rows = _loads(exclude_rows)

    df = _prepare_df_for_selection(file_id)

    from ..utils.helpers import _resolve_columns, _resolve_row_indices

    try:
        if mode == "columns":
            cols = _resolve_columns(df, sel_cols)
            df2 = df.loc[:, cols]
            if ex_cols:
                ex = _resolve_columns(df2, ex_cols)
                df2 = df2.drop(columns=ex)
        elif mode == "rows":
            rows_idx = _resolve_row_indices(df, sel_rows)
            df2 = df.iloc[rows_idx]
            if ex_rows:
                ex_idx = set(_resolve_row_indices(df, ex_rows))
                keep_positions = [i for i in rows_idx if i not in ex_idx]
                df2 = df.iloc[keep_positions]
        elif mode == "both":
            cols = _resolve_columns(df, sel_cols)
            rows_idx = _resolve_row_indices(df, sel_rows)
            df2 = df.iloc[rows_idx][cols]
            if ex_cols:
                ex = _resolve_columns(df2, ex_cols)
                df2 = df2.drop(columns=ex)
            if ex_rows:
                ex_idx = set(_resolve_row_indices(df, ex_rows))
                keep_positions = [i for i in rows_idx if i not in ex_idx]
                df2 = df.iloc[keep_positions][cols]
        else:
            raise HTTPException(status_code=400, detail="unknown mode")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Selection error ({type(e).__name__}): {e}") from e

    df2 = df2.fillna("")
    gen = dataframe_to_csv_stream(df2)
    headers = {"Content-Disposition": 'attachment; filename="parsed.csv"'}
    return StreamingResponse(gen(), media_type="text/csv", headers=headers)


@router.post(
    "/api/parse-download-format",
    tags=["Parsing"],
    summary="Parse, select and download in specified format",
    description="Select rows and/or columns from the uploaded file and download as CSV, XLSX, or both (ZIP).",
    responses={200: {"description": "File download response in specified format"}},
)
async def parse_download_format(
    file_id: str = Form(
        ...,
        description="File identifier returned from /api/upload-preview (e.g., 9f2d1c3e_sample.csv)",
    ),
    mode: str = Form(..., description="Selection mode: 'columns', 'rows', or 'both' (e.g., rows)"),
    format: str = Form("csv", description="Download format: 'csv', 'xlsx', or 'both' (both returns zip) (e.g., xlsx)"),
    has_header: bool = Form(True, description="Whether the first row contains headers. If false, first selected row will be used as header."),
    selected_columns: str | None = Form(
        None,
        description='JSON array of column names/indices to include (e.g., ["col1", "col3"])',
    ),
    selected_rows: str | None = Form(
        None,
        description="JSON array of row indices to include (e.g., [0, 2, 4])",
    ),
    exclude_columns: str | None = Form(
        None,
        description='JSON array of column names/indices to exclude (e.g., ["col2"])',
    ),
    exclude_rows: str | None = Form(
        None,
        description="JSON array of row indices to exclude (e.g., [1, 3])",
    ),
):
    """Download parsed data in specified format (CSV, XLSX, or both as ZIP)."""
    from ..utils.helpers import _prepare_df_for_selection

    def _loads(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    sel_cols = _loads(selected_columns)
    sel_rows = _loads(selected_rows)
    ex_cols = _loads(exclude_columns)
    ex_rows = _loads(exclude_rows)

    df = _prepare_df_for_selection(file_id)

    from ..utils.helpers import _resolve_columns, _resolve_row_indices

    try:
        if mode == "columns":
            cols = _resolve_columns(df, sel_cols)
            df2 = df.loc[:, cols]
            if ex_cols:
                ex = _resolve_columns(df2, ex_cols)
                df2 = df2.drop(columns=ex)
        elif mode == "rows":
            rows_idx = _resolve_row_indices(df, sel_rows)
            df2 = df.iloc[rows_idx]
            if ex_rows:
                ex_idx = set(_resolve_row_indices(df, ex_rows))
                keep_positions = [i for i in rows_idx if i not in ex_idx]
                df2 = df.iloc[keep_positions]
        elif mode == "both":
            cols = _resolve_columns(df, sel_cols)
            rows_idx = _resolve_row_indices(df, sel_rows)
            df2 = df.iloc[rows_idx][cols]
            if ex_cols:
                ex = _resolve_columns(df2, ex_cols)
                df2 = df2.drop(columns=ex)
            if ex_rows:
                ex_idx = set(_resolve_row_indices(df, ex_rows))
                keep_positions = [i for i in rows_idx if i not in ex_idx]
                df2 = df.iloc[keep_positions][cols]
        else:
            raise HTTPException(status_code=400, detail="unknown mode")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Selection error ({type(e).__name__}): {e}") from e

    # If has_header is False and we have rows selected, use first row as header
    if not has_header and len(df2) > 0:
        # Use first row as column names
        new_columns = [str(val) for val in df2.iloc[0].values]
        df2 = df2.iloc[1:].reset_index(drop=True)
        df2.columns = new_columns

    df2 = df2.fillna("")

    # Validate format
    if format not in ["csv", "xlsx", "both"]:
        raise HTTPException(status_code=400, detail="Invalid format. Must be 'csv', 'xlsx', or 'both'")

    # Return single CSV
    if format == "csv":
        gen = dataframe_to_csv_stream(df2)
        headers = {"Content-Disposition": 'attachment; filename="parsed.csv"'}
        return StreamingResponse(gen(), media_type="text/csv", headers=headers)

    # Return single XLSX
    elif format == "xlsx":
        output = io.BytesIO()
        df2.to_excel(output, index=False, sheet_name="Data")
        output.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="parsed.xlsx"'}
        return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

    # Return both as ZIP
    elif format == "both":
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add CSV
            csv_buffer = io.StringIO()
            df2.to_csv(csv_buffer, index=False)
            zip_file.writestr("parsed.csv", csv_buffer.getvalue())

            # Add XLSX
            xlsx_buffer = io.BytesIO()
            df2.to_excel(xlsx_buffer, index=False, sheet_name="Data")
            zip_file.writestr("parsed.xlsx", xlsx_buffer.getvalue())

        zip_buffer.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="parsed.zip"'}
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)
