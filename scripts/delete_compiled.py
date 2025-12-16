import traceback
from pathlib import Path

p = Path(r"d:\Projects\AST_Parser\backend_fastapi\test_outputs\2025_09_18_Wireless_Test_2_5G_Sampling_HH5K_Compiled.xlsx")
print("target", p)
try:
    if p.exists():
        print("exists, size=", p.stat().st_size)
        try:
            p.unlink()
            print("deleted")
        except Exception as e:
            print("delete exception:", type(e).__name__, e)
            traceback.print_exc()
    else:
        print("not found")
except Exception as e:
    print("error checking file:", type(e).__name__, e)
    traceback.print_exc()
