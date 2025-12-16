import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.utils.dvt_to_mc2_converter import (
    convert_dvt_to_mc2,
    determine_bands_from_rows,
    extract_serial_number,
    make_output_filename,
    parse_dvt_file,
)

router = APIRouter()

DVT_FILE = File(..., description="DVT format CSV file to convert")


@router.post(
    "/api/convert-dvt-to-mc2",
    tags=["DVT_MC2"],
    summary="Convert a DVT file to MC2 CSV",
    description="Upload a DVT CSV/XLSX file and receive a converted MC2-format CSV as an attachment.",
    responses={
        200: {
            "description": "MC2 CSV file returned as attachment",
            "content": {"text/csv": {"example": "ISN,Param1,Param2\n..."}},
        },
        400: {"description": "Bad request (parsing error)"},
        500: {"description": "Conversion failed"},
    },
)
async def convert_dvt_to_mc2_endpoint(dvt_file: UploadFile = DVT_FILE):
    """
    Convert a DVT format file to MC2 format.

    Accepts a DVT CSV file upload and returns the converted MC2 format as a CSV download.

    - **dvt_file**: The DVT format CSV file to convert
    - Returns: MC2 format CSV file as attachment
    """
    # Read uploaded file (support CSV and XLSX)
    try:
        raw = await dvt_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded DVT file: {e}") from e

    # If XLSX try to convert to CSV string first
    filename_lower = (dvt_file.filename or "").lower()
    if filename_lower.endswith((".xls", ".xlsx")):
        try:
            # Load sheet with no header to preserve raw rows exactly as in the file
            df = pd.read_excel(io.BytesIO(raw), sheet_name=0, dtype=str, header=None)
            # Export without pandas-generated header row so original rows are preserved
            dvt_text = df.to_csv(index=False, header=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse XLSX: {e}") from e
    else:
        try:
            dvt_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            # fallback: try latin-1
            dvt_text = raw.decode("latin-1", errors="ignore")

    try:
        # Parse DVT rows to extract serial and bands for filename
        dvt_rows = parse_dvt_file(dvt_text)
        serial = extract_serial_number(dvt_text)
        bands = determine_bands_from_rows(dvt_rows)
        out_name = make_output_filename(serial, bands)

        # Convert DVT to MC2 format (always include USL/LSL blank rows)
        mc2_csv_content = convert_dvt_to_mc2(dvt_text, include_usl_lsl=True)

        # Return CSV content with desired filename
        response = Response(
            content=mc2_csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}") from e
