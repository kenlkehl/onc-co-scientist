"""Exercise the paired reference harness against a running vLLM endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from report_smoke import write_report

from onc_co_scientist.expected_surprising.research import now
from onc_co_scientist.expected_surprising.rollout import run
from onc_co_scientist.expected_surprising.schemas import (
    PairSpec,
    ValidationPolicy,
    WorkflowVersions,
)
from onc_co_scientist.expected_surprising.summary import paired_summary
from onc_co_scientist.providers.registry import get_provider


class MeteredProvider:
    """Preserve the provider's behavior and retain response diagnostics separately."""

    def __init__(self, config, path, label):
        self.provider = get_provider(config)
        self.path, self.label, self.calls = path, label, 0

    @property
    def model_id(self):
        return self.provider.model_id

    def chat(self, messages, **kwargs):
        start = time.monotonic()
        self.calls += 1
        record = {
            "call": self.calls,
            "started_at": now(),
            "requested_max_tokens": kwargs.get("max_tokens"),
        }
        try:
            response = self.provider.chat(messages, **kwargs)
            if isinstance(response.raw, dict) and "metrics" in response.raw:
                record.update(response.raw["metrics"])
            else:
                choice = response.raw.choices[0]
                record.update(
                    finish_reason=choice.finish_reason,
                    response_chars=len(response.text),
                    reasoning_chars=len(getattr(choice.message, "reasoning_content", None) or ""),
                    usage=response.raw.usage.model_dump() if response.raw.usage else None,
                )
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_s"] = round(time.monotonic() - start, 3)
            with self.path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"{self.label}: call {self.calls}: {json.dumps(record)}", flush=True)


def summarize(report, out, iterations):
    transcript = [json.loads(x) for x in (out / "transcript.jsonl").read_text().splitlines()]
    stages = [x for x in transcript if x["kind"] == "stage"]
    results = [r for x in stages for r in x["results"]]
    analyses = [r for r in results if r["id"].startswith("analysis-")]
    validation = [r for r in results if r["id"].startswith("validation-")]
    last = report["exploration"][-1] if report["exploration"] else {}
    confirmation = report["confirmation"]
    scorable = bool(stages and any(r["valid"] for r in analyses))
    return {
        "run_id": report["run_id"],
        "profile": report["profile"],
        "version": report["version"],
        "completed": True,
        "scorable": scorable,
        "protocol_complete": len(stages) == 4 * iterations and not report["protocol_errors"],
        "protocol_clean": len(stages) == 4 * iterations and not report["attempt_errors"],
        "valid_stages": len(stages),
        "expected_stages": 4 * iterations,
        "protocol_errors": report["protocol_errors"],
        "attempt_errors": report["attempt_errors"],
        "recovered_stages": report["recovered_stages"],
        "hypotheses_proposed": last.get("cumulative_proposed", 0),
        "hypotheses_tested": last.get("cumulative_tested", 0),
        "valid_discovery_sample_analyses": sum(r["valid"] for r in analyses),
        "invalid_discovery_sample_analyses": sum(not r["valid"] for r in analyses),
        "validation_requests": len(report["validation"]["requests"]),
        "validation_deliveries": len(validation),
        "validation": report["validation"],
        "versions": report["versions"],
        "scores": report["scores"],
        "response_components": report["responsiveness"]["components"],
        "response_exclusions": report["responsiveness"]["exclusions"],
        "dataset_sha256": report["dataset_sha256"],
        "valid_validation_results": report["responsiveness"]["valid_n"],
        "responsiveness_accuracy": report["responsiveness"]["accuracy"],
        "post_evidence_windows": len(report["post_evidence_exploration"]),
        "accepted_claims": len(confirmation["results"]),
        "confirmed_target_discoveries": confirmation["confirmed_matches"],
        "additional_claims": confirmation["additional"],
        "discovery": report["discovery"],
        "primary_focal_recovery": confirmation["primary_recovery"],
        "first_attempt_primary_focal_recovery": confirmation["first_attempt_primary_recovery"],
        "report_path": str(out / "report.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data", type=Path, default=Path("data/expected_surprising_agent_judgment")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", default=["nsclc_clinical", "nsclc_depmap"])
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--replicate-id", default="smoke-replicate-0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=125000)
    parser.add_argument("--max-retries-per-stage", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = args.out / "api_metadata"
    metadata.mkdir()
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=20) as response:
        server = json.load(response)
    if args.model not in {m["id"] for m in server["data"]}:
        raise ValueError("Requested model is not served by this endpoint")
    policies = {
        profile: ValidationPolicy.default(profile, args.iterations).model_dump()
        for profile in args.profiles
    }
    config = {
        "versions": WorkflowVersions.model_validate(
            json.loads((args.data / "package_manifest.json").read_text())["versions"]
        ).model_dump(),
        "validation_policies": policies,
        "replicate_id": args.replicate_id,
        "provider": {
            "kind": "vllm_openai",
            "model_id": args.model,
            "base_url": args.base_url,
            "timeout_s": args.timeout_s,
        },
        "max_tokens_per_call": args.max_tokens,
        "max_retries_per_stage": args.max_retries_per_stage,
    }
    (args.out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "started_at": now(),
        "purpose": f"Protocol and scoring smoke test with {args.iterations} iterations per run.",
        "server": server,
        "config": config,
        "iterations": args.iterations,
        "workers": args.workers,
        "profiles": args.profiles,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("src/onc_co_scientist/expected_surprising").glob("*.py"))
        },
        "reporter_sha256": hashlib.sha256(
            Path(__file__).with_name("report_smoke.py").read_bytes()
        ).hexdigest(),
        "dependencies": {
            n: importlib.metadata.version(n)
            for n in ["openai", "numpy", "pandas", "scipy", "pydantic"]
        },
        "package_manifest_sha256": hashlib.sha256(
            (args.data / "package_manifest.json").read_bytes()
        ).hexdigest(),
        "tasks": [],
    }
    jobs = []
    for profile in args.profiles:
        paths = list((args.data / "private").glob(f"es-v2-{profile}-*/pair.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one pair for {profile}")
        path = paths[0]
        spec = PairSpec.model_validate_json(path.read_text())
        assignment = json.loads((path.parent / "assignment.json").read_text())
        for version in ("expected", "surprising"):
            run_id = f"{profile}-{version}"
            public = args.data / "public" / assignment[version]["task_id"]
            task = json.loads((public / "task.json").read_text())
            WorkflowVersions.model_validate(task["versions"])
            if task["versions"] != config["versions"]:
                raise ValueError("Task and manifest workflow versions differ")
            digest = hashlib.sha256((public / "dataset.parquet").read_bytes()).hexdigest()
            if digest != assignment[version]["sha256"]:
                raise ValueError("Public dataset differs from its recorded hash")
            jobs.append((spec, version, public, run_id))
            manifest["tasks"].append(
                {
                    "run_id": run_id,
                    "pair_id": spec.pair_id,
                    "version": version,
                    **assignment[version],
                    "spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def one(job):
        spec, version, public, run_id = job
        out = args.out / "runs" / run_id
        provider = MeteredProvider(config["provider"], metadata / f"{run_id}.jsonl", run_id)
        report = run(
            spec,
            version,
            public,
            out,
            provider,
            run_id=run_id,
            iterations=args.iterations,
            max_tokens_per_call=args.max_tokens,
            max_retries_per_stage=args.max_retries_per_stage,
            policy=policies[spec.profile],
            replicate_id=args.replicate_id,
        )
        return summarize(report, out, args.iterations)

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, job): job[-1] for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "run_id": futures[future],
                    "completed": False,
                    "scorable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            completed.append(result)
            completed.sort(key=lambda r: r["run_id"])
            (args.out / "summary.json").write_text(json.dumps(completed, indent=2))
            print("RUN COMPLETE: " + json.dumps(result), flush=True)
    reports = [
        json.loads((args.out / "runs" / job[-1] / "report.json").read_text())
        for job in jobs
        if (args.out / "runs" / job[-1] / "report.json").exists()
    ]
    paired = (
        paired_summary(reports)
        if len(reports) == len(jobs)
        else {"unavailable": "One or more assigned runs failed before writing a report"}
    )
    (args.out / "paired_summary.json").write_text(json.dumps(paired, indent=2))
    manifest["completed_at"] = now()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_report(args.out, args.out / "SMOKE_REPORT.md")


if __name__ == "__main__":
    main()
