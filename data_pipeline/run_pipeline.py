from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Books to Scrape -> SQLite pipeline")
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()

    command = [sys.executable, str(ROOT / "data_pipeline" / "scrape_books.py")]
    if args.offline_fixture:
        command.append("--offline-fixture")
    else:
        command.extend(["--pages", str(args.pages)])
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "data_pipeline" / "load_and_query.py")], cwd=ROOT, check=True)
    print("Data pipeline completed successfully.")


if __name__ == "__main__":
    main()
