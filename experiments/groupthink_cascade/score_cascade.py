#!/usr/bin/env python3
"""Recompute the cascade summary and report from saved successful records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade import write_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--out",
        type=Path,
        help="Combined output directory. Required when more than one result directory is supplied.",
    )
    args = parser.parse_args()
    if len(args.results) > 1 and args.out is None:
        raise SystemExit("--out is required when combining multiple result directories")
    output_dir = args.out or args.results[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    source_runs = []
    replicate_offset = 0
    for result_dir in args.results:
        source_records = json.loads((result_dir / "records.json").read_text(encoding="utf-8"))
        source_replicates = sorted({int(record["replicate"]) for record in source_records})
        remap = {
            old: replicate_offset + index for index, old in enumerate(source_replicates, start=1)
        }
        for record in source_records:
            copied = dict(record)
            copied["replicate"] = remap[int(record["replicate"])]
            records.append(copied)
        replicate_offset += len(source_replicates)
        source_runs.append(
            {
                "path": str(result_dir.resolve()),
                "records": len(source_records),
                "replicates": len(source_replicates),
            }
        )
    if len(args.results) > 1:
        (output_dir / "records.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "source_runs.json").write_text(
            json.dumps(source_runs, indent=2) + "\n", encoding="utf-8"
        )
    summary = write_summary(records, output_dir)
    print(summary["interpretation"])
    print(output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
