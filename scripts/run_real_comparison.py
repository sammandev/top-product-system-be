"""Run comparison on real sample files and write timestamped CSV and XLSX outputs.

Filename format: Golden_Compare_Compiled_YYYY_MM_DD_HHMMSSff.ext
where ff are first two digits of microseconds (centiseconds of a second).
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from app.utils.format_compare import (
    augment_for_human,
    compare_maps,
    format_minimal_number,
    parse_mastercontrol_text,
    parse_wifi_dvt_text,
    write_human_xlsx,
)

# ensure project root is on sys.path so 'backend_fastapi' imports work when script is run directly
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROOT = Path(__file__).resolve().parent.parent
sample_master = ROOT / "sample_data" / "DM2516770009406_MeasurePower_Lab_25G.csv"
sample_dvt = ROOT / "sample_data" / "Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv"
spec_path = ROOT / "spec_config.json"


def main():
    parser = argparse.ArgumentParser(description="Run real comparison and emit human CSV/XLSX")
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=None,
        help="Optional global margin threshold override (dB)",
    )
    args = parser.parse_args()

    if not sample_master.exists() or not sample_dvt.exists():
        print("Sample files not found; aborting")
        raise SystemExit(1)

    master_text = sample_master.read_text(errors="ignore")
    dvt_text = sample_dvt.read_text(errors="ignore")

    spec = None
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text())
        except Exception:
            spec = None

    master_map = parse_mastercontrol_text(master_text)
    dvt_map = parse_wifi_dvt_text(dvt_text)
    results = compare_maps(master_map, dvt_map, spec=spec)
    # pass runtime override into augment_for_human
    human_rows = augment_for_human(results, spec=spec, runtime_margin_threshold=args.margin_threshold)

    tz = timezone(timedelta(hours=7))
    # include timezone offset in filename; format microsecond then take first two digits as centiseconds
    filename_ts = datetime.now(tz).strftime("%Y_%m_%d_%H%M%S%f")
    ts_short = filename_ts[:-4]
    out_dir = ROOT / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"Golden_Compare_Compiled_{ts_short}.csv"
    xlsx_name = f"Golden_Compare_Compiled_{ts_short}.xlsx"
    csv_path = out_dir / csv_name
    xlsx_path = out_dir / xlsx_name

    # write CSV with minimal formatting consistent with augment_for_human
    prov = None
    if args.margin_threshold is not None:
        prov = {
            "margin_threshold": args.margin_threshold,
            "generated": datetime.now(UTC).isoformat(),
        }

    # write CSV rows first then append provenance as the last row (only if prov present)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # header first
        w.writerow(
            [
                "Antenna",
                "Mode",
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
            ant_label = int(r.get("antenna_dvt")) + 1 if r.get("antenna_dvt") is not None else ""
            metric_upper = str(r.get("metric", "")).upper()
            if metric_upper in ("POW", "EVM", "MASK", "FREQ", "LO_LEAKAGE_DB"):
                test_mode = "TX"
            elif metric_upper in ("PER", "RSSI"):
                test_mode = "RX"
            else:
                test_mode = "Others"
            mc2_val = format_minimal_number(r.get("mc2_value")) if r.get("mc2_value") is not None else "N/A"
            dvt_val = format_minimal_number(r.get("dvt_value")) if r.get("dvt_value") is not None else "N/A"
            w.writerow(
                [
                    ant_label,
                    test_mode,
                    r.get("metric"),
                    r.get("freq"),
                    r.get("standard"),
                    r.get("datarate"),
                    r.get("bandwidth"),
                    r.get("usl", "N/A"),
                    r.get("lsl", "N/A"),
                    mc2_val,
                    r.get("mc2_spec_diff", ""),
                    r.get("mc2_result", ""),
                    dvt_val,
                    r.get("dvt_spec_diff", ""),
                    r.get("dvt_result", ""),
                    r.get("mc2_dvt_diff", ""),
                ]
            )

        # provenance appended as final row only when runtime margin override was provided
        if prov is not None:
            w.writerow([f"PROVENANCE: margin_threshold={prov['margin_threshold']}; generated={prov['generated']} "])

    print("Wrote CSV to", csv_path)

    # write XLSX; pass provenance only when present
    if prov is not None:
        write_human_xlsx(human_rows, str(xlsx_path), provenance=prov, provenance_position="bottom")
    else:
        write_human_xlsx(human_rows, str(xlsx_path), provenance=None, provenance_position="bottom")
    print("Wrote XLSX to", xlsx_path)


if __name__ == "__main__":
    main()
