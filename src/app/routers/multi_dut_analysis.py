import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.utils.dut_criteria import load_criteria_from_bytes
from app.utils.multi_dut_analyzer import analyze_mc2_with_spec
from app.utils.spec_loader import load_spec_payload

router = APIRouter()

# Module-level default for spec_file to avoid calling File(...) directly in the function arguments
SPEC_FILE_REQUIRED = File(
    ...,
    description="Spec file (.json or .ini) describing measurement limits",
)

# Module-level default for mc2_file to avoid calling File(...) directly in the function arguments
MC2_FILE_DEFAULT = File(
    ...,
    description="MC2 CSV or XLSX file",
)


@router.post(
    "/api/analyze-multi-dut",
    tags=["MultiDUT"],
    summary="Analyze MC2 multi-DUT file and produce compiled XLSX",
    description="Upload a multi-DUT MC2 CSV/XLSX and optional spec JSON; returns an XLSX workbook with analysis.",
    responses={
        200: {
            "description": "Compiled XLSX workbook",
            "content": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {"example": "<binary xlsx content>"}},
        },
        400: {"description": "Spec missing or parse error"},
        500: {"description": "Analysis failed"},
    },
)
async def analyze_multi_dut(
    mc2_file: UploadFile = MC2_FILE_DEFAULT,
    spec_file: UploadFile = SPEC_FILE_REQUIRED,
):
    """Analyze an MC2 multi-DUT file using a provided JSON spec and return a compiled XLSX workbook."""
    try:
        raw = await mc2_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}") from e

    filename_lower = (mc2_file.filename or "").lower()
    # Convert XLSX to CSV if needed
    if filename_lower.endswith((".xls", ".xlsx")):
        try:
            df = pd.read_excel(io.BytesIO(raw), sheet_name=0, dtype=str, header=0)
            mc2_text = df.to_csv(index=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse XLSX: {e}") from e
    else:
        try:
            mc2_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            mc2_text = raw.decode("latin-1", errors="ignore")

    # Load spec content (JSON or INI)
    try:
        spec_bytes = await spec_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read spec file: {e}") from e
    spec = None
    criteria_rules = []
    try:
        spec_payload = load_spec_payload(spec_bytes, getattr(spec_file, "filename", None))
        spec = spec_payload.json_spec
    except ValueError:
        try:
            criteria_map = load_criteria_from_bytes(spec_bytes)
            if not criteria_map:
                raise ValueError("criteria file contained no rules")
            for bucket in criteria_map.values():
                if bucket:
                    criteria_rules.extend(bucket)
        except Exception as ini_exc:
            raise HTTPException(status_code=400, detail=f"Unable to parse spec file: {ini_exc}") from ini_exc
    if spec is None and not criteria_rules:
        raise HTTPException(status_code=400, detail="Spec file did not provide usable configuration")

    try:
        xlsx_bytes, summary = analyze_mc2_with_spec(
            mc2_text,
            spec,
            criteria_rules=criteria_rules,
        )
        out_name = mc2_file.filename or "analysis"
        if out_name.lower().endswith(".csv"):
            out_name = out_name[:-4]
        out_name = out_name + "_Compiled.xlsx"

        return Response(
            content=xlsx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e
