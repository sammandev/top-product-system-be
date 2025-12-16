"""Compare MasterControlV2 and WiFi DVT CSV test outputs.

Produces a CSV `compare_output.csv` with columns: antenna,metric,master_value,dvt_value,diff

Usage:
  python scripts/compare_formats.py --master <master.csv> --dvt <dvt.csv>
"""

import argparse
import json
from datetime import UTC
from pathlib import Path

from app.utils.format_compare import (
    augment_for_human,
    compare_maps,
    compute_summary,
)
from app.utils.format_compare import (
    parse_mastercontrol_text as parse_mastertext,
)
from app.utils.format_compare import (
    parse_wifi_dvt_text as parse_dvttext,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", required=True)
    p.add_argument("--dvt", required=True)
    p.add_argument("--threshold", type=float, default=None, help="Flag diffs >= threshold")
    p.add_argument(
        "--freq-tol",
        type=float,
        default=2.0,
        help="Frequency tolerance in MHz for fuzzy matching",
    )
    p.add_argument("--out-csv", default="compare_output.csv")
    p.add_argument("--out-json", default="compare_summary.json")
    p.add_argument("--out-xlsx", default=None, help="Optional human XLSX output path")
    p.add_argument(
        "--human",
        action="store_true",
        help="Write a human-friendly CSV with Pass/Fail text and labeled antennas",
    )
    p.add_argument(
        "--margin-threshold",
        type=float,
        default=None,
        help="Optional runtime margin threshold override (dB)",
    )
    args = p.parse_args()

    # If user didn't set a custom out-csv and human is requested, generate a timestamped filename in UTC+7
    if args.human and (args.out_csv == "compare_output.csv" or not args.out_csv):
        from datetime import datetime, timedelta, timezone

        tz = timezone(timedelta(hours=7))
        ts = datetime.now(tz).strftime("%Y_%m_%d_%H%M%S%f")
        ts_short = ts[:-4]
        args.out_csv = f"Golden_Compare_Compiled_{ts_short}.csv"

    master_text = Path(args.master).read_text()
    dvt_text = Path(args.dvt).read_text()

    master_map = parse_mastertext(master_text)
    dvt_map = parse_dvttext(dvt_text)

    # try to load spec_config.json from repo root if present
    spec_path = Path(__file__).resolve().parent.parent / "spec_config.json"
    spec = None
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text())
        except Exception:
            spec = None

    results = compare_maps(
        master_map,
        dvt_map,
        threshold=args.threshold,
        spec=spec,
        freq_tolerance_mhz=args.freq_tol,
    )
    human_rows = augment_for_human(results, spec=spec, runtime_margin_threshold=args.margin_threshold)

    # write CSV with enhanced human format
    import csv

    def compute_spec_limits(spec, standard, idx, metric_key):
        """Return (usl, lsl) for a given metric in the spec. None if not applicable."""
        if not spec or not standard:
            return (None, None)
        s_for = spec.get(standard)
        if not s_for:
            return (None, None)
        vals = s_for.get(metric_key)
        if not vals or idx is None or idx >= len(vals):
            return (None, None)
        rule = vals[idx]
        if rule is None or str(rule).strip() == "":
            return (None, None)
        # For EVM and MASK metrics these are USL-only
        # For FREQ and LO_LEAKAGE_DB the spec may be a range or a single number
        return rule

    prov = None
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if args.human:
            # New requested header and order; include provenance row only when runtime override provided
            from datetime import datetime

            if args.margin_threshold is not None:
                prov = {
                    "margin_threshold": args.margin_threshold,
                    "generated": datetime.now(UTC).isoformat(),
                }
                w.writerow([f"PROVENANCE: margin_threshold={prov['margin_threshold']}; generated={prov['generated']} "])
            w.writerow(
                [
                    "Test Mode",
                    "Antenna",
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
        else:
            w.writerow(
                [
                    "antenna_dvt",
                    "antenna_mc2",
                    "metric",
                    "freq",
                    "standard",
                    "datarate",
                    "bandwidth",
                    "mc2_value",
                    "dvt_value",
                    "diff",
                    "flag",
                ]
            )

        for r in human_rows if args.human else results:
            ant_dvt = r.get("antenna_dvt")
            ant_mc2 = r.get("antenna_mc2")
            ant_label = int(ant_dvt) + 1 if ant_dvt is not None else ""
            # determine Test Mode
            metric_upper = str(r.get("metric", "")).upper()
            if metric_upper in ("POW", "EVM", "MASK", "FREQ", "LO_LEAKAGE_DB"):
                test_mode = "TX"
            elif metric_upper in ("PER", "RSSI"):
                test_mode = "RX"
            else:
                test_mode = "Others"

            # write human row using values computed by augment_for_human
            if args.human:
                # use minimal formatting for numeric values
                from app.utils.format_compare import (
                    format_minimal_number,
                )

                mc2_val_disp = format_minimal_number(r.get("mc2_value")) if r.get("mc2_value") is not None else "N/A"
                dvt_val_disp = format_minimal_number(r.get("dvt_value")) if r.get("dvt_value") is not None else "N/A"
                w.writerow(
                    [
                        test_mode,
                        ant_label,
                        r["metric"],
                        r["freq"],
                        r["standard"],
                        r["datarate"],
                        r["bandwidth"],
                        r.get("usl", "N/A"),
                        r.get("lsl", "N/A"),
                        mc2_val_disp,
                        r.get("mc2_spec_diff", ""),
                        r.get("mc2_result", ""),
                        dvt_val_disp,
                        r.get("dvt_spec_diff", ""),
                        r.get("dvt_result", ""),
                        r.get("mc2_dvt_diff", ""),
                    ]
                )
            else:
                # keep old behavior for non-human
                w.writerow(
                    [
                        ant_dvt,
                        ant_mc2,
                        r["metric"],
                        r["freq"],
                        r["standard"],
                        r["datarate"],
                        r["bandwidth"],
                        r.get("mc2_value"),
                        r.get("dvt_value"),
                        f"{float(r.get('diff', 0)):.2f}",
                        int(bool(r["flag"])),
                    ]
                )

    # write JSON summary grouped by DVT antenna as Ant_1..Ant_N and include aggregates
    summary = {}
    for r in results:
        key = f"Ant_{int(r['antenna_dvt']) + 1}"
        summary.setdefault(key, []).append(r)
    aggregates = compute_summary(results)
    out_obj = {
        "groups": summary,
        "aggregates": aggregates,
        "meta": {"freq_tolerance_mhz": args.freq_tol},
    }
    Path(args.out_json).write_text(json.dumps(out_obj, indent=2))

    # print a small human-readable summary (per-antenna counts and flagged count)
    agg = {}
    for r in results:
        a = r["antenna_dvt"]
        agg.setdefault(a, {"total": 0, "flagged": 0})
        agg[a]["total"] += 1
        if r["flag"]:
            agg[a]["flagged"] += 1

    print(f"Wrote {len(results)} comparisons to {args.out_csv} and summary to {args.out_json}")
    print("Per-antenna summary:")
    for a in sorted(agg.keys()):
        print(f"  antenna_dvt={a}: {agg[a]['flagged']}/{agg[a]['total']} flagged")

    # optionally write XLSX
    # if user didn't provide an XLSX path but requested human output and wants xlsx, create timestamped filename
    if args.human and args.out_xlsx is None:
        from datetime import datetime, timedelta, timezone

        tz = timezone(timedelta(hours=7))
        ts = datetime.now(tz).strftime("%Y_%m_%d_%H%M%S%f")
        ts_short = ts[:-4]
        args.out_xlsx = f"Golden_Compare_Compiled_{ts_short}.xlsx"

    if args.out_xlsx and args.human:
        try:
            from app.utils.format_compare import write_human_xlsx

            # write to path or to BytesIO depending on provided arg
            try:
                from io import BytesIO

                bio = BytesIO()
                write_human_xlsx(human_rows, bio, provenance=prov, provenance_position="bottom")
                # if user provided a path, save bytes
                if args.out_xlsx:
                    Path(args.out_xlsx).write_bytes(bio.getvalue())
                    print(f"Wrote human XLSX to {args.out_xlsx}")
            except Exception:
                # fallback: write to path directly
                write_human_xlsx(
                    human_rows,
                    args.out_xlsx,
                    provenance=prov,
                    provenance_position="bottom",
                )
                print(f"Wrote human XLSX to {args.out_xlsx}")
        except Exception as e:
            print(f"Failed to write XLSX: {e}")


if __name__ == "__main__":
    main()
