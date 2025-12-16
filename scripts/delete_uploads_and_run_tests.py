import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"d:\Projects\AST_Parser")
BACKEND = ROOT / "backend_fastapi"
UPLOADS = BACKEND / "uploads"


def remove_uploads():
    if UPLOADS.exists():
        try:
            # remove directory tree
            shutil.rmtree(UPLOADS)
            print(f"removed {UPLOADS}")
        except Exception as e:
            print("failed to remove uploads:", e)
            return False
    else:
        print("uploads folder not found, nothing to remove")
    return True


def run_pytest():
    # run pytest with the same interpreter executing this script
    cmd = [sys.executable, "-m", "pytest", "backend_fastapi/tests", "-q"]
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.Popen(cmd, env=env)
    p.wait()
    return p.returncode


if __name__ == "__main__":
    ok = remove_uploads()
    if not ok:
        print("aborting tests due to delete failure")
        sys.exit(2)
    rc = run_pytest()
    print("pytest exit code:", rc)
    sys.exit(rc)
