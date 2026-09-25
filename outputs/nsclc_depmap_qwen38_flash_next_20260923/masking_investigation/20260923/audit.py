"""Offline audit of the frozen DepMap label transform and observed actions."""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "source/src"))

import pandas as pd
from onc_co_scientist.expected_surprising.schemas import PairSpec, Hypothesis
from onc_co_scientist.expected_surprising.masking import SemanticMask, MaskedValidationService
from onc_co_scientist.expected_surprising.evaluation import WorkflowValidationService
from onc_co_scientist.expected_surprising.generation import version_discoveries
from onc_co_scientist.expected_surprising.scoring import (
    claim_key, comparison_key, estimate, match, workflow_discovery,
)

PRIVATE = ROOT / "input_data/unmasked/private/es-v2-nsclc_depmap-42005"
MP = ROOT / "input_data/masked/private/es-v2-nsclc_depmap-42005"
PAIR = PairSpec.model_validate_json((PRIVATE / "pair.json").read_text())
MASK = SemanticMask(json.loads((MP / "masking.json").read_text()))
POLICY = json.loads((PRIVATE / "workflow.json").read_text())["policy"]
SOURCE = Path(json.loads((OUT / "source.json").read_text())["original_snapshot"])


def state(snapshot):
    hs, tests = {}, {}
    for e in snapshot["stages"]:
        rec = e["record"]
        for raw in rec["hypotheses"]:
            h = Hypothesis.model_validate(raw)
            hs[h.id] = h
        if rec["stage"] == "analyze":
            for hid, result in zip(rec["executed_ids"], e["results"], strict=True):
                if result["valid"]:
                    tests.setdefault(comparison_key(hs[hid]), hs[hid])
    accepted_keys = {claim_key(hs[hid]) for hid in snapshot["stages"][-1]["record"]["accepted_ids"]}
    unique = {}
    for h in hs.values():
        if claim_key(h) in accepted_keys:
            unique.setdefault(claim_key(h), h)
    return hs, tests, list(unique.values())


def main():
    frames = {}
    for version in ("expected", "surprising"):
        for label, private in (("unmasked", PRIVATE), ("masked", MP)):
            assignment = json.loads((private / "assignment.json").read_text())
            frames[label, version] = pd.read_parquet(ROOT / "input_data" / label / "public" / assignment[version]["task_id"] / "dataset.parquet")
        pd.testing.assert_frame_equal(MASK.frame(frames["unmasked", version]), frames["masked", version], check_exact=True)
    assert (PRIVATE / "pair.json").read_bytes() == (MP / "pair.json").read_bytes()
    checks = {"exact_dataframe_parity": True, "same_private_pair": True, "renamed_predictors_and_outcomes": len(MASK.columns), "counterfactual_runs": [], "numeric_comparisons_checked": 0, "validation_seeds_checked": 0}
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for index, item in enumerate(manifest["snapshots"], 1):
        snap = json.loads((SOURCE / item["run_id"] / "snapshot.json").read_text())
        old = json.loads((SOURCE / item["run_id"] / "interim_report.json").read_text())
        hs, tests, accepted = state(snap)
        inverse = item["masking"] == "masked"
        other = "unmasked" if inverse else "masked"
        convert = lambda h: MASK.hypothesis(h, inverse=inverse)
        services = {
            "unmasked": WorkflowValidationService(PAIR, item["version"], snap["header"]["replicate_id"], POLICY),
            "masked": MaskedValidationService(PAIR, item["version"], snap["header"]["replicate_id"], POLICY, MASK),
        }
        original_service, transformed_service = services[item["masking"]], services[other]
        transformed_tests = {comparison_key(convert(h)): convert(h) for h in tests.values()}
        for h in tests.values():
            h2 = convert(h)
            assert MASK.hypothesis(h2, inverse=not inverse) == h
            delta = next(o.delta for o in original_service.spec.outcomes if o.name == h.outcome)
            a = estimate(frames[item["masking"], item["version"]], h, delta=delta, alpha=.05, result_id="parity")
            b = estimate(frames[other, item["version"]], h2, delta=delta, alpha=.05, result_id="parity")
            assert a == b, (item["run_id"], h.id, "numerics")
            assert original_service.derive_seed("independent-validation", comparison_key(h)) == transformed_service.derive_seed("independent-validation", comparison_key(h2))
            checks["numeric_comparisons_checked"] += 1
            checks["validation_seeds_checked"] += 1
        confirmation = transformed_service.confirm([convert(h) for h in accepted])
        score = workflow_discovery([convert(h) for h in accepted], version_discoveries(transformed_service.spec, item["version"]), transformed_tests, confirmation, failed=False, repaired=bool(snap["attempt_errors"]))
        for key in ("exact", "exact_or_near", "accepted_n", "tested_accepted_n", "confirmed_tested_n", "independently_supported_n", "unsupported_acceptance_n", "unconfirmed_n"):
            assert score[key] == old["discovery"][key], (item["run_id"], key)
        assert confirmation["results"] == old["confirmation"]["results"], (item["run_id"], "confirmation")
        checks["counterfactual_runs"].append({"run_id": item["run_id"], "direction": item["masking"] + " to " + other, "exact_D": score["exact"]["D"], "all_scores_unchanged": True})
        print(json.dumps({"renaming_checks_completed": index, "numerical_comparisons": checks["numeric_comparisons_checked"]}), flush=True)
    oracle = []
    for version in ("expected", "surprising"):
        named = WorkflowValidationService(PAIR, version, "replicate-001", POLICY)
        opaque = MaskedValidationService(PAIR, version, "replicate-001", POLICY, MASK)
        truths = [d.hypothesis for d in version_discoveries(PAIR, version)]
        opaque_truths = [MASK.hypothesis(h) for h in truths]
        a = named.confirm(truths); b = opaque.confirm(opaque_truths)
        assert a["results"] == b["results"]
        sa = workflow_discovery(truths, version_discoveries(PAIR, version), {comparison_key(h) for h in truths}, a)
        sb = workflow_discovery(opaque_truths, version_discoveries(opaque.spec, version), {comparison_key(h) for h in opaque_truths}, b)
        assert sa["exact"] == sb["exact"]
        validation = []
        for i, (h, mh) in enumerate(zip(truths, opaque_truths, strict=True), 1):
            na, _ = named.deliver(h, i, "voluntary"); ma, _ = opaque.deliver(mh, i, "voluntary")
            assert na.model_dump(exclude={"id"}) == ma.model_dump(exclude={"id"})
            validation.append(h.id)
        oracle.append({"version": version, "named_D": sa["exact"]["D"], "masked_D": sb["exact"]["D"], "named_focal": a["primary_recovery"], "masked_focal": b["primary_recovery"], "validation_pairs_checked": len(validation)})
    checks["oracle_checks"] = oracle
    (OUT / "parity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    behavior = []
    for item in manifest["snapshots"]:
        snap = json.loads((SOURCE / item["run_id"] / "snapshot.json").read_text())
        hs = {}; valid = {}; stage1 = {}; executed = 0; invalid = 0; repeats = 0; id_uses = 0
        outcomes = Counter(); first = []
        for event in snap["stages"]:
            rec = event["record"]
            if rec["iteration"] > 6: continue
            for raw in rec["hypotheses"]:
                h = Hypothesis.model_validate(raw)
                if item["masking"] == "masked": h = MASK.hypothesis(h, inverse=True)
                hs[h.id] = h
                id_uses += int(h.exposure == "cell_line_id" or any(c.variable == "cell_line_id" for c in h.eligibility + h.subgroup))
            if rec["stage"] != "analyze": continue
            for hid, result in zip(rec["executed_ids"], event["results"], strict=True):
                h = hs[hid]; executed += 1; outcomes[h.outcome] += 1
                if rec["iteration"] == 1: first.append(h.model_dump())
                if not result["valid"]: invalid += 1; continue
                key = comparison_key(h); repeats += int(key in valid); valid.setdefault(key, h)
                if rec["iteration"] == 1: stage1.setdefault(key, h)
        targets = version_discoveries(PAIR, item["version"])
        coverage = lambda candidates: [d.hypothesis.id for d in targets if any(match(h, d.hypothesis, ignore_direction=True) == "exact" for h in candidates)]
        behavior.append({"run_id": item["run_id"], "masking": item["masking"], "version": item["version"], "executed": executed, "invalid": invalid, "repeated": repeats, "unique_valid": len(valid), "identifier_proposals": id_uses, "first_iteration_targets_tested": coverage(stage1.values()), "six_iteration_targets_tested": coverage(valid.values()), "first_iteration_comparisons_unmasked": first, "tested_outcomes": dict(outcomes), "attempt_errors": sum(e['iteration'] <= 6 for e in snap["attempt_errors"])})
    aggregates = []
    for version in ("expected", "surprising"):
        for label in ("unmasked", "masked"):
            rows = [x for x in behavior if x["version"] == version and x["masking"] == label]
            aggregates.append({"version": version, "masking": label, "runs": len(rows), **{key: sum(x[key] for x in rows) for key in ("executed", "invalid", "repeated", "unique_valid", "identifier_proposals", "attempt_errors")}, "first_iteration_target_test_counts": dict(Counter(t for x in rows for t in x["first_iteration_targets_tested"])), "six_iteration_target_test_counts": dict(Counter(t for x in rows for t in x["six_iteration_targets_tested"]))})
    (OUT / "behavior.json").write_text(json.dumps({"checkpoint": 6, "runs": behavior, "cells": aggregates}, indent=2) + "\n")
    print(json.dumps({"parity": {k: v for k, v in checks.items() if k != "counterfactual_runs"}, "behavior_cells": aggregates}, indent=2), flush=True)


if __name__ == "__main__": main()
