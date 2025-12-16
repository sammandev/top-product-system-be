"""Smoke script to generate a human XLSX using the new writer for manual verification."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from backend_fastapi.app.utils.format_compare import (
    augment_for_human,
    compare_maps,
    write_human_xlsx,
)
from backend_fastapi.app.utils.format_compare import (
    parse_mastercontrol_text as parse_master,
)
from backend_fastapi.app.utils.format_compare import (
    parse_wifi_dvt_text as parse_dvt,
)

OUT = Path(__file__).resolve().parent.parent / "test_outputs"
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Smoke generate human XLSX")
parser.add_argument("--margin-threshold", type=float, default=None)
args = parser.parse_args()

filename_ts = datetime.now(UTC).strftime("%Y_%m_%d_%H%M%S%f")
ts_short = filename_ts[:-4]
out_path = OUT / f"Golden_Compare_Compiled_smoke_{ts_short}.xlsx"

master_text = "TX1_POW_2412_11AC_CCK11_B20,19.6\n"
dvt_text = "Standard,Antenna,Freq,DataRate,BandWidth,M. Power\n11AC,0,2412,CCK11,B20,19.4\n"

master_map = parse_master(master_text)
dvt_map = parse_dvt(dvt_text)
spec = {
    "11AC": {
        "frequency": [2412],
        "bw": ["B20"],
        "mod": ["CCK11"],
        "tx_target_power": [20],
        "tx_target_tolerance": ["-0.5~0.5"],
    }
}
res = compare_maps(master_map, dvt_map, spec=spec)
aug = augment_for_human(res, spec=spec, runtime_margin_threshold=args.margin_threshold)
prov = None
if args.margin_threshold is not None:
    prov = {
        "margin_threshold": args.margin_threshold,
        "generated": datetime.now(UTC).isoformat(),
    }
write_human_xlsx(aug, str(out_path), provenance=prov, provenance_position="bottom")
print("Wrote", out_path)
