"""Refresh progress from saved transcripts without changing running experiments."""

import argparse
import json
import time
from pathlib import Path

from run_replicates import write_status

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
out = args.out.resolve()
manifest = json.loads((out / "manifest.json").read_text())
while True:
    summaries = json.loads((out / "summary.json").read_text())
    results = {r["run_id"]: r for r in summaries}
    states = {}
    for job in manifest["jobs"]:
        run_id = job["run_id"]
        if run_id in results:
            state = "finished" if results[run_id]["completed"] else "failed"
        elif (out / "runs" / run_id / "transcript.jsonl").exists():
            state = "running"
        else:
            state = "queued"
        states[run_id] = {"status": state}
    write_status(out, manifest, states)
    execution = json.loads((out / "execution.json").read_text())
    if execution.get("completed_at"):
        break
    time.sleep(30)
