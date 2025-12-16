"""Run multi-DUT analysis and write to a new filename to avoid locked file issues."""

import datetime
import json
from pathlib import Path

from app.utils.multi_dut_analyzer import analyze_mc2_with_spec

mc2_path = Path(r"d:\Projects\AST_Parser\backend_fastapi\sample_data\2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv")
spec_path = Path(r"d:\Projects\AST_Parser\backend_fastapi\sample_data_config\multi-dut_all_specs.json")

mc2_text = mc2_path.read_text(encoding="utf-8", errors="ignore")
spec = json.loads(spec_path.read_text(encoding="utf-8"))

xlsx_bytes, summary = analyze_mc2_with_spec(mc2_text, spec)

out_dir = Path(r"d:\Projects\AST_Parser\backend_fastapi\test_outputs")
out_dir.mkdir(parents=True, exist_ok=True)
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = out_dir / f"2025_09_18_Wireless_Test_2_5G_Sampling_HH5K_Compiled_{now}.xlsx"
out.write_bytes(xlsx_bytes)
print("Wrote", out)
print("SUMMARY:", summary)
