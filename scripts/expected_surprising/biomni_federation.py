"""Prepare, validate and explicitly launch native Biomni federation experiments."""

import argparse
import json
from pathlib import Path

from onc_co_scientist.external.federation_runner import execute, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "run"])
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.grid)
    else:
        result = execute(
            args.grid.resolve() / "biomni", pilot=args.command == "pilot", resume=args.resume
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
