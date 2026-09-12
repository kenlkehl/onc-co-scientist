"""Archive a throttled attempt and replay from its earliest affected stage."""

import hashlib
import json
import shutil
from pathlib import Path

STAGES = ("explore", "analyze", "appraise", "synthesize")


def position(request):
    return int(request["iteration"]), STAGES.index(request["stage"])


def recover(run_dir):
    run_dir = Path(run_dir)
    report = run_dir / "run.json"
    result = json.loads(report.read_text()) if report.exists() else {}
    if result.get("status") == "completed":
        return dict(status="retained_completed", run_dir=str(run_dir))
    archive = run_dir / "recovery/azure_rate_limits"
    manifest = archive / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    if archive.exists():
        raise RuntimeError(f"Incomplete recovery requires inspection: {archive}")
    records = []
    failures = []
    for path in sorted((run_dir / "calls").rglob("*.json")):
        record = json.loads(path.read_text())
        request = record.get("request", record)
        boundary = position(request)
        records.append((path, boundary))
        error = str(record.get("result", {}).get("error", "")).lower()
        if "429" in error or "rate limit" in error or "rate_limit" in error:
            failures.append(boundary)
    if not failures:
        if result.get("scientific_report_sha256"):
            raise RuntimeError(f"Terminal non-throttling failure needs inspection: {run_dir}")
        return dict(status="resume_checkpoint", run_dir=str(run_dir))
    boundary = min(failures)
    paths = [path for path, at in records if at >= boundary]
    paths += [p for p in (report, run_dir / "report.json") if p.exists()]
    # Handoffs are derived outputs, never replay inputs. Preserve them as a unit.
    paths += [p for p in (run_dir / "handoffs").rglob("*") if p.is_file()]
    entries = [
        dict(path=str(p.relative_to(run_dir)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in paths
    ]
    value = dict(
        status="rolled_back",
        run_dir=str(run_dir),
        boundary=dict(iteration=boundary[0], stage=STAGES[boundary[1]]),
        files=entries,
        reason="Retry throttling without dependent cached responses",
    )
    archive.mkdir(parents=True)
    (archive / "intent.json").write_text(json.dumps(value, indent=2))
    for entry in entries:
        old = run_dir / entry["path"]
        new = archive / entry["path"]
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(old, new)
    manifest.write_text(json.dumps(value, indent=2))
    return value
