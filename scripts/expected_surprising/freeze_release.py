"""Retain reproducible DGP specifications and a compact audit of a local data release."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

from onc_co_scientist.expected_surprising.evaluation import REFERENCE_POLICY_VERSION
from onc_co_scientist.expected_surprising.generation import version_discoveries
from onc_co_scientist.expected_surprising.review_policy import (
    REVIEW_POLICY_VERSION,
    require_current_pair,
)
from onc_co_scientist.expected_surprising.schemas import PairSpec

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=ROOT / "data/expected_surprising_v2")
parser.add_argument("--out", type=Path, default=ROOT / "benchmarks/expected_surprising/v2")
args = parser.parse_args()
DATA, OUT = args.data, args.out
(OUT / "specs").mkdir(parents=True, exist_ok=True)
(OUT / "calibration").mkdir(exist_ok=True)
entries = []
for path in sorted((DATA / "private").glob("*/pair.json")):
    spec = PairSpec.model_validate_json(path.read_text())
    require_current_pair(spec)
    cal_path = path.parent / "calibration.json"
    calibration = json.loads(cal_path.read_text()) if cal_path.exists() else None
    if (
        not calibration
        or not calibration.get("calibration_passed")
        or calibration.get("reference_policy_version") != REFERENCE_POLICY_VERSION
        or calibration.get("replicates", 0) < 1000
        or calibration.get("pair_id") != spec.pair_id
    ):
        raise ValueError(f"Current passing calibration required before freezing {spec.pair_id}")
    compact = spec.model_dump()
    for evidence in compact["evidence"]:
        for source in evidence["sources"]:
            # Raw retrievals remain in the local private research audit. The
            # source ID, title, DOI, URL, scope decision, and rationale travel
            # with the versioned specification needed for replay.
            source["abstract"] = ""
            source["full_text_excerpt"] = ""
    spec_path = OUT / "specs" / f"{spec.pair_id}.json"
    spec_path.write_text(json.dumps(compact, indent=2) + "\n")
    assignment = json.loads((path.parent / "assignment.json").read_text())
    if calibration:
        (OUT / "calibration" / f"{spec.pair_id}.json").write_text(
            json.dumps(calibration, indent=2) + "\n"
        )
    focal = next(d for d in spec.discoveries if d.hypothesis.id == spec.focal_id)
    source = next(e for e in spec.evidence if e.candidate.hypothesis.id == focal.evidence_id)
    events_path = DATA / "research" / spec.profile / "events.jsonl"
    events = [json.loads(x) for x in events_path.read_text().splitlines()]
    entries.append(
        {
            "pair_id": spec.pair_id,
            "profile": spec.profile,
            "n_per_version": spec.n,
            "focal_mode": spec.focal_mode,
            "focal_id": spec.focal_id,
            "discovery_counts": {
                version: dict(Counter(d.category for d in version_discoveries(spec, version)))
                for version in ("expected", "surprising")
            },
            "expected_statement": source.candidate.statement,
            "expected_direction": focal.expected_direction,
            "literature_ids": source.review.source_ids,
            "assignments": assignment,
            "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "calibration_passed": bool(calibration and calibration["calibration_passed"]),
            "calibration_replicates": calibration["replicates"] if calibration else 0,
            "effect_relative_difference": calibration["effect_relative_difference"]
            if calibration
            else None,
            "snr_relative_difference": calibration["snr_relative_difference"]
            if calibration
            else None,
            "reference_recovery": calibration["reference_recovery"] if calibration else None,
            "proposed_candidate_rejections": sum(e["kind"] == "rejected" for e in events),
            "targeted_generation_guidance": [
                {k: v for k, v in e.items() if k != "kind"}
                for e in events
                if e["kind"] == "targeted_generation_guidance"
            ],
            "expert_review_complete": False,
            "review_policy_version": REVIEW_POLICY_VERSION,
            "candidate_realism_passed": True,
            "full_dgp_realism_passed": True,
            "status": "development",
        }
    )
code = {}
for folder in ("synthetic", "expected_surprising"):
    for path in sorted((ROOT / "src/onc_co_scientist" / folder).rglob("*.py")):
        code[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "release": "expected-surprising-v2",
    "review_policy_version": REVIEW_POLICY_VERSION,
    "reference_policy_version": REFERENCE_POLICY_VERSION,
    "status": "development",
    "python": platform.python_version(),
    "dependencies": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "pyarrow", "pydantic")
    },
    "pair_n": len(entries),
    "task_n": 2 * len(entries),
    "source_sha256": code,
    "pairs": entries,
}
(OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
lines = [
    "# Expected / surprising development release",
    "",
    f"{len(entries)} paired datasets; {2 * len(entries)} task instances. "
    "Each contains six embedded discoveries.",
    "",
    "The specifications preserve literature review decisions and citation metadata. "
    "Full retrievals and model exchanges remain in `data/expected_surprising_v2/research/`. "
    "Expert adjudication is pending.",
    "",
    "| Profile | Rows / version | Focal comparison | Mode | Calibration |",
    "|---|---:|---|---|---|",
]
for e in entries:
    label = (
        "Passed (1,000 replicates)"
        if e["calibration_passed"]
        else f"Development ({e['calibration_replicates']} replicates)"
    )
    lines.append(
        f"| {e['profile']} | {e['n_per_version']:,} | {e['focal_id']} | "
        f"{e['focal_mode']} | {label} |"
    )
lines += [
    "",
    "Replay the frozen specifications with:",
    "",
    "```bash",
    "ocs expected-surprising materialize benchmarks/expected_surprising/v2/specs "
    "data/expected_surprising_replay",
    "```",
    "",
    "Use the dependency versions in `release_manifest.json` for identical Parquet serialization. "
    "The source hashes record the generation implementation. "
    "Expose only the assigned public task to an agent.",
]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print(OUT / "release_manifest.json")
