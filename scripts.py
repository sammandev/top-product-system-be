import subprocess
import sys


def format_code():
    subprocess.run(["uv", "run", "ruff", "format", "."])


def lint():
    subprocess.run(["uv", "run", "ruff", "check", "."])


def lint_fix():
    subprocess.run(["uv", "run", "ruff", "check", "--fix"])


def check():
    subprocess.run(["uv", "run", "ruff", "check", "."])
    subprocess.run(["uv", "run", "ruff", "format", "--check", "."])
    subprocess.run(["uv", "run", "ruff", "check", "--fix", "."])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts.py [lint|format|fix|check]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "format":
        format_code()
    elif command == "lint":
        lint()
    elif command == "fix":
        lint_fix()
    elif command == "check":
        check()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
