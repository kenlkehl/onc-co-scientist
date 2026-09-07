"""Resume re-review, replacement, materialization and calibration of the ten profiles."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from onc_co_scientist.expected_surprising import cli
from onc_co_scientist.expected_surprising.evaluation import REFERENCE_POLICY_VERSION
from onc_co_scientist.expected_surprising.review_policy import (
    REVIEW_POLICY_VERSION,
    require_current_pair,
)
from onc_co_scientist.expected_surprising.schemas import PairSpec
from onc_co_scientist.synthetic.cancer_types import all_cancer_types

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profiles", nargs="*", default=[str(c) for c in all_cancer_types()])
parser.add_argument("--workers", type=int, default=2)
parser.add_argument(
    "--config", type=Path, default=ROOT / "configs/expected_surprising.example.yaml"
)
args = parser.parse_args()
out = ROOT / "data/expected_surprising_v2"
specs = ROOT / "benchmarks/expected_surprising/v1/specs"


def rebuild(profile):
    audit = out / "research" / profile
    report = audit / "recheck.json"
    previous = json.loads(report.read_text()) if report.exists() else {}
    if (
        len(previous.get("results", [])) != 6
        or previous.get("policy_version") != REVIEW_POLICY_VERSION
    ):
        checkpoint = audit / f"{profile}_reviewed.json"
        proposed = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
        cli.recheck(args.config, specs, out, profile)
        retained = json.loads(checkpoint.read_text())
        ids = {r["candidate"]["hypothesis"]["id"] for r in retained}
        retained += [r for r in proposed if r["candidate"]["hypothesis"]["id"] not in ids]
        checkpoint.write_text(json.dumps(retained, indent=2))
    print(f"{profile}: recheck complete", flush=True)
    cli.research(args.config, out, profile)
    print(f"{profile}: candidate inventory complete", flush=True)
    pairs = list((out / "private").glob(f"es-v2-{profile}-*/pair.json"))
    if not pairs:
        cli.generate(args.config, out, profile)
        pairs = list((out / "private").glob(f"es-v2-{profile}-*/pair.json"))
    if len(pairs) != 1:
        raise ValueError(f"Expected one pair for {profile}")
    pair = pairs[0]
    require_current_pair(PairSpec.model_validate_json(pair.read_text()))
    cal = pair.parent / "calibration.json"
    calibration = json.loads(cal.read_text()) if cal.exists() else {}
    if (
        not calibration.get("calibration_passed")
        or calibration.get("reference_policy_version") != REFERENCE_POLICY_VERSION
    ):
        cli.calibrate_command(pair, replicates=1000, reference_n=100000)
    if not json.loads(cal.read_text())["calibration_passed"]:
        raise ValueError(f"Calibration failed for {profile}")
    return str(pair)


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(rebuild, p): p for p in args.profiles}
    results = {}
    for future in as_completed(futures):
        profile = futures[future]
        try:
            results[profile] = {"complete": True, "pair": future.result()}
        except Exception as exc:
            results[profile] = {"complete": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({profile: results[profile]}), flush=True)
        # Each invocation has its own report so concurrent bounded batches do
        # not overwrite one another's progress.
        name = "_".join(args.profiles)
        (out / f"rebuild_{name}.json").write_text(json.dumps(results, indent=2) + "\n")
