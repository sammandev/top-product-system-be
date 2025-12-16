import json
from datetime import UTC, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from app.utils.format_compare import (
    augment_for_human,
    compare_maps,
    compute_summary,
    format_minimal_number,
    parse_mastercontrol_text,
    parse_wifi_dvt_text,
    write_human_xlsx,
)

router = APIRouter()


# Module-level File() dependency objects to avoid calling File() in argument defaults
_UPLOAD_REQUIRED = File(...)
_UPLOAD_OPTIONAL_NONE = File(None)


@router.post(
    "/api/compare-formats",
    tags=["Comparison"],
    summary="Compare mc2 and DVT formatted data",
    description="Upload a MasterControl file and a DVT file to compare formats. Optionally return human-friendly CSV or XLSX.",
    responses={
        200: {
            "description": "Comparison result or downloadable CSV/XLSX when requested",
            "content": {
                "application/json": {
                    "example": {
                        "rows": [{"metric": "POW", "mc2_value": "1.23"}],
                        "summary": {"pass": 5},
                    }
                },
                "text/csv": {"example": "Antenna,Metric,Freq,MC2 Value\n1,POW,2412,1.23"},
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {"example": "<binary xlsx>"},
            },
        }
    },
)
async def compare_formats(
    master_file: UploadFile = _UPLOAD_REQUIRED,
    dvt_file: UploadFile = _UPLOAD_REQUIRED,
    threshold: float | None = Form(
        None,
        description="Threshold value for comparison tolerance (e.g., 0.5)",
    ),
    margin_threshold: float | None = Form(
        None,
        description="Margin threshold override for pass/fail determination (e.g., 1.0)",
    ),
    spec_file: UploadFile | None = _UPLOAD_OPTIONAL_NONE,
    freq_tol: float | None = Form(
        2.0,
        description="Frequency tolerance in MHz for matching entries (e.g., 2.0)",
    ),
    human: bool | None = Form(
        False,
        description="Return human-readable CSV/XLSX format instead of JSON (e.g., true)",
    ),
    return_xlsx: bool | None = Form(
        False,
        description="Return XLSX file instead of CSV (only when human=True) (e.g., true)",
    ),
    request: Request = None,
):
    try:
        raw_master = await master_file.read()
        raw_dvt = await dvt_file.read()
        master_text = raw_master.decode("utf-8", errors="ignore")
        dvt_text = raw_dvt.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}") from e

    # fallback: if the dvt upload read zero bytes, try to inspect the full request form
    # (some test clients may wire multipart data oddly). Use request to find any uploaded
    # file parts and try to recover the DVT bytes.
    try:
        if not raw_dvt and isinstance(request, Request):
            form = await request.form()
            # prefer the named field 'dvt_file'
            candidate = None
            if "dvt_file" in form:
                candidate = form.get("dvt_file")
            else:
                # pick first UploadFile in form values with non-empty filename
                for v in form.values():
                    if hasattr(v, "filename") and v.filename:
                        candidate = v
                        break
            if candidate is not None:
                try:
                    # candidate may be an UploadFile with async read
                    try:
                        alt_raw = await candidate.read()
                    except Exception:
                        # fall back to reading from file attr
                        candidate.file.seek(0)
                        alt_raw = candidate.file.read()
                    if alt_raw:
                        raw_dvt = alt_raw
                        dvt_text = raw_dvt.decode("utf-8", errors="ignore")
                except Exception:
                    pass
    except Exception:
        # non-fatal; continue and let parser report
        pass

    try:
        mc2_map = parse_mastercontrol_text(master_text)
        try:
            dvt_map = parse_wifi_dvt_text(dvt_text)
        except Exception as pe:
            # Provide a small diagnostic preview of the uploaded DVT content so tests
            # can determine why the parser failed to detect the title row. Limit to
            # the first 12 CSV rows to keep the message compact.
            try:
                import csv
                from io import StringIO as _StringIO

                rdr = csv.reader(_StringIO(dvt_text))
                rows = list(rdr)[:12]
                preview = "\n".join(",".join(r) for r in rows)
            except Exception:
                preview = "<unable to generate preview>"
            # if preview is empty, also include a short repr and length of the decoded text
            extra = ""
            try:
                if not preview:
                    rd_len = len(raw_dvt) if "raw_dvt" in locals() else "<no raw>"
                    md_len = len(raw_master) if "raw_master" in locals() else "<no raw>"
                    md_name = getattr(master_file, "filename", "<no name>")
                    dv_name = getattr(dvt_file, "filename", "<no name>")
                    extra = (
                        f"\nDVT text length={len(dvt_text)}; snippet={repr(dvt_text[:400])}; "
                        f"raw_bytes_len={rd_len}; raw_bytes_snippet={repr(raw_dvt[:200] if 'raw_dvt' in locals() else b'')}; "
                        f"master_raw_len={md_len}; master_filename={md_name}; dvt_filename={dv_name}"
                    )
            except Exception:
                extra = ""
            raise HTTPException(
                status_code=400,
                detail=(f"Compare error: DVT parse failed: {pe}. DVT preview (first 12 rows):\n{preview}{extra}"),
            ) from pe
        # try to load spec_config.json from repo root unless user uploaded one
        spec = None
        try:
            if spec_file is not None:
                spec_text = (await spec_file.read()).decode("utf-8", errors="ignore")
                spec = json.loads(spec_text)
            else:
                spec_path = Path(__file__).resolve().parents[2] / "spec_config.json"
                if spec_path.exists():
                    spec = json.loads(spec_path.read_text())
        except Exception:
            spec = None

        results = compare_maps(
            mc2_map,
            dvt_map,
            threshold=threshold,
            spec=spec,
            freq_tolerance_mhz=float(freq_tol or 2.0),
        )
        # augment for human-friendly fields; pass runtime margin threshold override
        human_rows = augment_for_human(results, spec=spec, runtime_margin_threshold=margin_threshold)
        summary = compute_summary(human_rows)
        if human:
            # produce CSV text (using minimal formatting) and optionally XLSX
            import csv

            buf = StringIO()
            w = csv.writer(buf)

            from datetime import datetime

            prov = None
            if margin_threshold is not None:
                prov = {
                    "margin_threshold": margin_threshold,
                    "generated": datetime.now(UTC).isoformat(),
                }
                w.writerow([f"PROVENANCE: margin_threshold={prov['margin_threshold']}; generated={prov['generated']} "])
            w.writerow(
                [
                    "Antenna",
                    "Test Mode",
                    "Metric",
                    "Freq",
                    "Standard",
                    "DataRate",
                    "BW",
                    "USL",
                    "LSL",
                    "MC2 Value",
                    "MC2 & Spec Diff",
                    "MC2 Result",
                    "DVT Value",
                    "DVT & Spec Diff",
                    "DVT Result",
                    "MC2 & DVT Diff",
                ]
            )
            for r in human_rows:
                ant_label = int(r["antenna_dvt"]) + 1 if r.get("antenna_dvt") is not None else ""
                mc2_val = format_minimal_number(r.get("mc2_value")) if r.get("mc2_value") is not None else "N/A"
                dvt_val = format_minimal_number(r.get("dvt_value")) if r.get("dvt_value") is not None else "N/A"
                # ensure the spec diffs are strings already formatted by augment_for_human; otherwise format
                mc2_spec = r.get("mc2_spec_diff", "")
                dvt_spec = r.get("dvt_spec_diff", "")
                # determine Test Mode
                metric_upper = str(r.get("metric", "")).upper()
                if metric_upper in ("POW", "EVM", "MASK", "FREQ", "LO_LEAKAGE_DB"):
                    test_mode = "TX"
                elif metric_upper in ("PER", "RSSI"):
                    test_mode = "RX"
                else:
                    test_mode = "Others"

                w.writerow(
                    [
                        ant_label,
                        test_mode,
                        r["metric"],
                        r["freq"],
                        r["standard"],
                        r["datarate"],
                        r["bandwidth"],
                        r.get("usl", "N/A"),
                        r.get("lsl", "N/A"),
                        mc2_val,
                        mc2_spec,
                        r.get("mc2_result", ""),
                        dvt_val,
                        dvt_spec,
                        r.get("dvt_result", ""),
                        r.get("mc2_dvt_diff", ""),
                    ]
                )

            csv_text = buf.getvalue()
            # if XLSX requested, produce bytes and return as application/vnd...
            if return_xlsx:
                # write XLSX to BytesIO using updated write_human_xlsx
                bio = BytesIO()
                write_human_xlsx(human_rows, bio, provenance=prov, provenance_position="bottom")
                try:
                    bio.seek(0)
                except Exception:
                    pass
                data = bio.getvalue()
                # filename with UTC+7 timezone
                tz = timezone(timedelta(hours=7))
                filename_ts = datetime.now(tz).strftime("%Y_%m_%d_%H%M%S%f")
                filename = f"Golden_Compare_Compiled_{filename_ts}.xlsx"
                return Response(
                    content=data,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )

            # otherwise return CSV text
            tz = timezone(timedelta(hours=7))
            filename_ts = datetime.now(tz).strftime("%Y_%m_%d_%H%M%S%f")
            filename = f"Golden_Compare_Compiled_{filename_ts}.csv"
            return Response(
                content=csv_text,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        # for programmatic API return augmented rows so clients have USL/LSL and diffs
        return JSONResponse({"rows": human_rows, "summary": summary})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Compare error: {e}") from e
