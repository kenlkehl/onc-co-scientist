#!/usr/bin/env python3
"""Recompute summary.json and report.md from a completed pilot records file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilot import write_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    records_path = args.results_dir / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    summary = write_summary(records, args.results_dir)
    print(summary["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
