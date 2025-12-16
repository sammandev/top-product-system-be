import json

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils.helpers import (
    _compare_dfs,
    _loads_join,
    _prepare_df_for_selection,
    dataframe_to_csv_stream,
)

router = APIRouter()


@router.post(
    "/api/compare",
    tags=["Comparison"],
    summary="Compare two uploaded datasets",
    description="Compare two previously uploaded files (by file_id) using selection parameters and return a JSON diff.",
    responses={200: {"description": "Comparison payload (rows/summary)"}},
)
async def compare(
    file_a: str = Form(..., description="First file identifier (from /api/upload-preview) (e.g., 9f2d1c3e_A.csv)"),
    file_b: str = Form(..., description="Second file identifier (from /api/upload-preview) (e.g., 4b7d8e2f_B.csv)"),
    mode: str = Form(..., description="Comparison mode: 'columns', 'rows', or 'both' (e.g., both)"),
    a_selected_columns: str | None = Form(None, description='JSON array of columns to select from file A (e.g., ["ISN", "Result"])'),
    a_selected_rows: str | None = Form(None, description="JSON array of row indices to select from file A (e.g., [0, 5, 10])"),
    a_exclude_columns: str | None = Form(None, description='JSON array of columns to exclude from file A (e.g., ["Timestamp"])'),
    a_exclude_rows: str | None = Form(None, description="JSON array of row indices to exclude from file A (e.g., [2, 3])"),
    b_selected_columns: str | None = Form(None, description='JSON array of columns to select from file B (e.g., ["ISN", "Result"])'),
    b_selected_rows: str | None = Form(None, description="JSON array of row indices to select from file B (e.g., [1, 4, 7])"),
    b_exclude_columns: str | None = Form(None, description='JSON array of columns to exclude from file B (e.g., ["Operator"])'),
    b_exclude_rows: str | None = Form(None, description="JSON array of row indices to exclude from file B (e.g., [0])"),
    a_join_on: str | None = Form(None, description='JSON array or string of column(s) to join on in file A (e.g., ["ISN"])'),
    b_join_on: str | None = Form(None, description='JSON array or string of column(s) to join on in file B (e.g., ["ISN"])'),
    request: Request = None,
):
    if request is not None and not __import__("os").environ.get("ASTPARSER_API_KEY"):
        # keep existing behavior (API key optional)
        pass

    def _loads(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    a_sel_cols = _loads(a_selected_columns)
    a_sel_rows = _loads(a_selected_rows)
    a_ex_cols = _loads(a_exclude_columns)
    a_ex_rows = _loads(a_exclude_rows)

    b_sel_cols = _loads(b_selected_columns)
    b_sel_rows = _loads(b_selected_rows)
    b_ex_cols = _loads(b_exclude_columns)
    b_ex_rows = _loads(b_exclude_rows)

    # parse join-on fields which might be JSON-encoded by the client
    a_join_on_parsed = _loads_join(a_join_on)
    b_join_on_parsed = _loads_join(b_join_on)

    df_a = _prepare_df_for_selection(file_a)
    df_b = _prepare_df_for_selection(file_b)

    try:
        payload = _compare_dfs(
            df_a,
            df_b,
            mode,
            a_sel_cols,
            a_sel_rows,
            a_ex_cols,
            a_ex_rows,
            b_sel_cols,
            b_sel_rows,
            b_ex_cols,
            b_ex_rows,
            a_join_on=a_join_on_parsed,
            b_join_on=b_join_on_parsed,
        )
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Compare error ({type(e).__name__}): {e}") from e


@router.post(
    "/api/compare-download",
    tags=["Comparison"],
    summary="Compare and download CSV",
    description="Perform comparison and download results as CSV.",
    responses={200: {"description": "CSV download of compare results"}},
)
async def compare_download(
    file_a: str = Form(..., description="First file identifier (from /api/upload-preview) (e.g., 9f2d1c3e_A.csv)"),
    file_b: str = Form(..., description="Second file identifier (from /api/upload-preview) (e.g., 4b7d8e2f_B.csv)"),
    mode: str = Form(..., description="Comparison mode: 'columns', 'rows', or 'both' (e.g., columns)"),
    a_selected_columns: str | None = Form(None, description='JSON array of columns to select from file A (e.g., ["ISN", "Result"])'),
    a_selected_rows: str | None = Form(None, description="JSON array of row indices to select from file A (e.g., [0, 5, 10])"),
    a_exclude_columns: str | None = Form(None, description='JSON array of columns to exclude from file A (e.g., ["Timestamp"])'),
    a_exclude_rows: str | None = Form(None, description="JSON array of row indices to exclude from file A (e.g., [2, 3])"),
    b_selected_columns: str | None = Form(None, description='JSON array of columns to select from file B (e.g., ["ISN", "Result"])'),
    b_selected_rows: str | None = Form(None, description="JSON array of row indices to select from file B (e.g., [1, 4, 7])"),
    b_exclude_columns: str | None = Form(None, description='JSON array of columns to exclude from file B (e.g., ["Operator"])'),
    b_exclude_rows: str | None = Form(None, description="JSON array of row indices to exclude from file B (e.g., [0])"),
    a_join_on: str | None = Form(None, description='JSON array or string of column(s) to join on in file A (e.g., ["ISN"])'),
    b_join_on: str | None = Form(None, description='JSON array or string of column(s) to join on in file B (e.g., ["ISN"])'),
    request: Request = None,
):
    def _loads(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    a_sel_cols = _loads(a_selected_columns)
    a_sel_rows = _loads(a_selected_rows)
    a_ex_cols = _loads(a_exclude_columns)
    a_ex_rows = _loads(a_exclude_rows)

    b_sel_cols = _loads(b_selected_columns)
    b_sel_rows = _loads(b_selected_rows)
    b_ex_cols = _loads(b_exclude_columns)
    b_ex_rows = _loads(b_exclude_rows)

    a_join_on_parsed = _loads_join(a_join_on)
    b_join_on_parsed = _loads_join(b_join_on)

    df_a = _prepare_df_for_selection(file_a)
    df_b = _prepare_df_for_selection(file_b)

    try:
        payload = _compare_dfs(
            df_a,
            df_b,
            mode,
            a_sel_cols,
            a_sel_rows,
            a_ex_cols,
            a_ex_rows,
            b_sel_cols,
            b_sel_rows,
            b_ex_cols,
            b_ex_rows,
            a_join_on=a_join_on_parsed,
            b_join_on=b_join_on_parsed,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Compare error ({type(e).__name__}): {e}") from e

    df_out = __import__("pandas").DataFrame(payload.get("rows", []))
    gen = dataframe_to_csv_stream(df_out)
    headers = {"Content-Disposition": 'attachment; filename="compare.csv"'}
    return StreamingResponse(gen(), media_type="text/csv", headers=headers)
