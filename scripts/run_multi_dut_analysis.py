"""Quick script to run multi-DUT analysis locally for smoke testing."""

import json
from pathlib import Path

from app.utils.multi_dut_analyzer import analyze_mc2_with_spec

mc2_path = Path(r"d:\Projects\AST_Parser\backend_fastapi\sample_data\2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv")
spec_path = Path(r"d:\Projects\AST_Parser\backend_fastapi\sample_data_config\multi-dut_all_specs.json")

mc2_text = mc2_path.read_text(encoding="utf-8", errors="ignore")
spec = spec_path.read_text(encoding="utf-8")

spec = json.loads(spec)

xlsx_bytes, summary = analyze_mc2_with_spec(mc2_text, spec)

out = Path(r"d:\Projects\AST_Parser\backend_fastapi\test_outputs\2025_09_18_Wireless_Test_2_5G_Sampling_HH5K_Compiled.xlsx")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(xlsx_bytes)
print("Wrote", out, "summary=", summary)
