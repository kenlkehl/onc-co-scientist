"""Freeze, execute, monitor and score the balanced 120-run Gemini federation grid."""
from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import UTC, datetime
from pathlib import Path

from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text

PROFILE = "gemini38_flash_medium"
MODEL = "gemini-3.8-flash"
WORKFLOWS = ("persistent", "sequential", "deliberative")
CONDITIONS = ("expected-unmasked", "expected-masked", "surprising-unmasked", "surprising-masked")


def read(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except FileNotFoundError:
        return default


def now():
    return datetime.now(UTC).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_grid(grid, model=MODEL):
    cells = collections.Counter((r["workflow_id"], r["condition"], r["semantic_condition"]) for r in grid)
    expected = {(w,c,s) for w in WORKFLOWS for c in ("named","masked") for s in ("expected","surprising")}
    if set(cells) != expected or set(cells.values()) != {10} or len(grid) != 120:
        raise ValueError("Require exactly 12 cells with 10 repeats each")
    if len({(r["condition"], r["run_id"]) for r in grid}) != 120:
        raise ValueError("Duplicate scientific run identity")
    for key in expected:
        rows = [r for r in grid if (r["workflow_id"],r["condition"],r["semantic_condition"]) == key]
        if {r["replicate"] for r in rows} != set(range(1,11)):
            raise ValueError("Each cell requires repeats 1 through 10")
    if any(r["site_count"] != 4 or r["model_id"] != model or r["iterations"] != 25 for r in grid):
        raise ValueError("Unexpected site count, model or iteration budget")


def prepare(root, predecessor, backend="gemini"):
    local = backend == "gemma"
    model = "gemma4-31b" if local else MODEL
    profile = "gemma_4_31b" if local else PROFILE
    label = "Gemma 4 31B" if local else "Gemini 3.8 Flash"
    import yaml
    import pandas as pd
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans
    from onc_co_scientist.expected_surprising.site_statistics import partition_frame
    from onc_co_scientist.expected_surprising.federation_spec import FederationCell

    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_text(root / "LIVE_PROGRESS.md", f"# {label}: four-site federation\n\nPreparing 120 experiments: 12 cells × 10 repeats.\n")
    shutil.copytree(repo / "src", root / "source/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script_dir = root / "source/scripts/expected_surprising"
    script_dir.mkdir(parents=True)
    for name in ("gemini_federation_grid.py", "build_presentation_results.py", "report_federation_initial30.py", "presentation_costs.py"):
        shutil.copy2(repo / "scripts/expected_surprising" / name, script_dir / name)
    shutil.copy2(repo / "pyproject.toml", root / "source/pyproject.toml")
    grid, partitions = [], {}
    for condition in ("named", "masked"):
        folder = root / condition
        folder.mkdir()
        shutil.copytree(predecessor / condition / "input_data", folder / "input_data")
        config = yaml.safe_load((predecessor / condition / "config.yaml").read_text())
        config.update(experiment_id=f"{root.name}_{condition}", output_root=str(folder),
                      description=f"{label}; four random sites; 25 iterations; balanced 10-repeat grid",
                      replicates=10, max_parallel=12)
        config["models"] = [dict(id=profile, model_id=model, adapter="provider", reasoning_effort="medium",
            provider_config=dict(kind="gemini_vertex", model_id=MODEL, project_id="profile-notes",
                location="global", timeout_s=300, max_retries=8, reasoning_effort="medium"))]
        if local:
            config["models"][0]["provider_config"] = dict(kind="vllm_openai", model_id=model,
                base_url="http://camus:8060/v1", timeout_s=1800, max_retries=5, reasoning_effort="medium",
                sampling_profile="gemma4", json_object_output=True, disable_thinking_on_final_retry=True)
        config["workflows"] = [dict(id=w,mode=w,**(dict(agents_per_stage=2,deliberation_rounds=1) if w=="deliberative" else {})) for w in WORKFLOWS]
        config["federation"] = dict(site_counts=[4], seed=20260908,
            partitions=[dict(id="random",mode="random")], context_policy="federated_v2")
        config["expected_surprising"].update(root=str(folder / "input_data"), max_tokens_per_call=65536,
            max_retries_per_stage=2, persistent_history_chars=120000,
            peer_failure_policy="chair_with_available", stage_failure_policy="retain_scientific_scores")
        config["budget"].update(max_agent_calls=4500, max_runtime_seconds_per_call=1800)
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        spec = load_experiment_spec(folder / "config.yaml")
        rows = [dict(iterations=spec.iteration_policy.iterations, **p.public_dict()) for p in build_run_plans(spec)]
        atomic_write_json(folder / "plan.json", rows)
        grid.extend(dict(condition=condition, **r) for r in rows)
        for task in spec.tasks:
            frame = pd.read_parquet(task.public_workspace / "dataset.parquet")
            for repeat in range(1,11):
                _, audit = partition_frame(frame, FederationCell(sites=4, seed=20260908),
                    task.metadata["pair_id"], f"replicate-{repeat:03d}")
                key = f"r{repeat:03d}"
                if key in partitions and partitions[key] != audit:
                    raise ValueError("Condition presentations must share paired site partitions")
                partitions[key] = audit
    grid.sort(key=lambda r:(r["replicate"],WORKFLOWS.index(r["workflow_id"]),r["semantic_condition"],r["condition"]))
    verify_grid(grid, model)
    atomic_write_json(root / "grid.json", grid)
    atomic_write_json(root / "partition_audit.json", partitions)
    atomic_write_json(root / "selections" / f"{profile}.json", [dict(condition=r["condition"],run_id=r["run_id"]) for r in grid])
    atomic_write_json(root / "release_policy.json", dict(released_models=[profile], held_models=[],
        instruction=f"User authorized all 120 {label} experiments and final scoring, 2026-09-20."))
    paths = [p for base in (root/"source",root/"named/input_data",root/"masked/input_data") for p in base.rglob("*") if p.is_file()]
    paths += [root/c/"config.yaml" for c in ("named","masked")] + [root/"grid.json",root/"partition_audit.json"]
    atomic_write_json(root / "frozen_manifest.json", dict(created_at=now(), project=None if local else "profile-notes",
        model=model, label=label, backend=backend, endpoint="http://camus:8060/v1" if local else "Vertex global", source_predecessor=str(predecessor), hashes={str(p.relative_to(root)):sha(p) for p in paths},
        pricing=(dict(basis="Local endpoint; infrastructure and electricity not estimated") if local else dict(input_per_million=.75,cached_input_per_million=.075,output_per_million=3.75,
            valid_until="2026-12-31",source="https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing"))))
    refresh(root)


def refresh(root, cache=None):
    cache = {} if cache is None else cache
    metadata = read(root / "frozen_manifest.json", {})
    label = metadata.get("label", "Gemini 3.8 Flash")
    local = metadata.get("backend") == "gemma"
    grid = read(root / "grid.json", [])
    state = read(root / "execution.json", {})
    active = {(r["condition"],r["run_id"]) for r in state.get("active",[])}
    stopped = {(r["condition"],r["run_id"]):r for r in state.get("finished",[])}
    totals = collections.Counter()
    cells = collections.defaultdict(collections.Counter)
    detail = []
    for row in grid:
        folder = root / row["condition"] / "runs" / row["run_id"]
        ident = row["condition"],row["run_id"]
        result = read(folder/"run.json", {})
        status = result.get("status") or ("active" if ident in active else stopped.get(ident,{}).get("status","queued"))
        condition = row["semantic_condition"] + "-" + ("unmasked" if row["condition"]=="named" else "masked")
        counts = cells[row["workflow_id"],condition]
        counts[status] += 1
        usage = collections.Counter()
        latest = None
        for p in (folder/"provider_audit").glob("call-*/attempt-*/metrics.json"):
            if str(p) not in cache:
                cache[str(p)] = read(p)
            m = cache[str(p)]
            usage.update(m["usage"])
            usage["estimated_cost_usd"] += m["estimated_cost_usd"]
            usage["responses"] += 1
            latest = max(latest or "",m["ended_at"])
        for key,value in usage.items():
            totals[key] += value
            counts[key] += value
        # Saved central forms indicate stage activity, not guaranteed committed science.
        central = list((folder/"calls/central").glob("i*.json"))
        progress = max(central,key=lambda p:p.stat().st_mtime).stem if central else "waiting for first central response"
        if status == "active":
            activities = list((folder/"provider_audit").glob("call-*/activity.json"))
            if activities:
                activity = read(max(activities,key=lambda p:p.stat().st_mtime),{})
                progress += "; " + activity.get("status","")
        detail.append(dict(workflow=row["workflow_id"],condition=condition,repeat=row["replicate"],
            status=status,progress=progress,last_response=latest,source=str(folder),**usage))
    statuses = collections.Counter(r["status"] for r in detail)
    snapshot = dict(updated_at=now(),driver=state.get("status","prepared"),counts=dict(statuses),totals=dict(totals),runs=detail)
    atomic_write_json(root/"progress.json",snapshot)
    lines=[f"# {label}: four-site federation","",f"Updated {snapshot['updated_at']}","",
        ("Endpoint: **camus:8060**" if local else "GCP project: **profile-notes**") + " · Medium reasoning · Four random sites · 25 iterations · 10 repeats per cell.","",
        f"**Run status:** {dict(statuses)}. Driver: {snapshot['driver']}.","",
        f"Recorded input tokens: {totals['input_tokens']:,}; output tokens (including reasoning): {totals['output_tokens']:,}; cached input: {totals['cached_input_tokens']:,}.",
        ("Local endpoint: no per-token API charge is assumed; infrastructure and electricity costs are not estimated." if local else f"Estimated inference cost from saved receipts: **${totals['estimated_cost_usd']:,.2f}**. Interrupted requests without usage receipts and the minimal connection test are excluded."),"",
        "| Workflow | Condition | Completed | Active | Queued | Failed / interrupted | Responses | Estimated cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for w in WORKFLOWS:
        for c in CONDITIONS:
            x=cells[w,c]
            lines.append(f"| {w} | {c} | {x['completed']}/10 | {x['active']} | {x['queued']} | {x['failed']+x['interrupted']} | {x['responses']} | ${x['estimated_cost_usd']:.2f} |")
    lines += ["",f"[Scientific scores]({root/'analysis/REPORT.md'}) — generated automatically when the queue finishes.","",
        "## Individual runs","","Stage labels below show the latest saved central response; completion is verified from the final run record.","",
        "| Workflow | Condition | Repeat | Status | Latest activity | Last response (UTC) |","|---|---|---:|---|---|---|"]
    for r in detail:
        lines.append(f"| {r['workflow']} | {r['condition']} | {r['repeat']} | {r['status']} | {r['progress']} | {r['last_response'] or '—'} |")
    atomic_write_text(root/"LIVE_PROGRESS.md","\n".join(lines)+"\n")
    return snapshot


def score(root):
    from scripts.expected_surprising import build_presentation_results as scoring
    from scripts.expected_surprising.report_federation_initial30 import aggregate_sites, markdown_table
    metadata=read(root/"frozen_manifest.json",{})
    label=metadata.get("label","Gemini 3.8 Flash")
    local=metadata.get("backend")=="gemma"
    rows=[]
    for plan in read(root/"grid.json"):
        row=scoring.inspect_job((plan,"unmasked" if plan["condition"]=="named" else "masked",root/plan["condition"]))
        row.update(site_count=4,partition_id="random",harness="vLLM" if local else "Gemini Vertex")
        if row["report_available"]:
            report=read(Path(row["source"])/"report.json")
            row["input_tokens"]=report["coordination"]["usage"].get("input_tokens")
            row["successful_stages"]=report["successful_stages"]
            for cls in scoring.CLASSES:
                component=report["responsiveness"]["components"][cls]
                row["response_"+cls+"_correct"]=component["numerator"]
                row["response_"+cls+"_eligible"]=component["denominator"]
        rows.append(row)
    out=root/"analysis";out.mkdir(exist_ok=True)
    scoring.write_csv(out/"run_metrics.csv",rows)
    lines=[f"# {label}: scientific results","",f"Generated {now()}","",
        "Four random sites; 25 iterations; medium reasoning; 12 cells × 10 repeats. Same frozen synthetic datasets and existing scientific scoring definitions.","",
        "All scores are 0–100. P = precision; R = category-balanced recall; F1* = scientific F1; E = exploration; B = evidence responsiveness; Focal = independently confirmed exact focal recovery; C = weighted condition score."]
    all_tables={}
    for cohort in ("completed","finished"):
        table,cells,cross=aggregate_sites(rows,cohort)
        for name,data in (("grouped_metrics",table),("condition_metrics",cells),("cross_condition_scores",cross)):
            scoring.write_csv(out/f"{cohort}_{name}.csv",data)
        all_tables[cohort]=dict(groups=table,conditions=cells,cross=cross)
        lookup={(r['workflow'],r['condition'],r['metric']):r for r in cells}
        display=[]
        for w in WORKFLOWS:
            for c in CONDITIONS:
                selected=[r for r in rows if r['workflow']==w and r['version']+'-'+r['view']==c]
                display.append([w,c,f"{sum(r['status']=='completed' for r in selected)}/10",*[lookup.get((w,c,m),{}).get('value') for m in ('precision','recall','f1','exploration','responsiveness','focal_recovery','condition_score')]])
        lines += ["",f"## {'Completed runs' if cohort=='completed' else 'All terminal reports, including failed runs'}","",
            markdown_table(['Workflow','Condition','Completed','P','R','F1*','E','B','Focal','C'],display),"",
            markdown_table(['Workflow','Metric','Value','Availability'],[[r['workflow'],r['metric'],r['value'],r['value_status']] for r in cross])]
    progress=refresh(root)
    lines += ["","## Token use and scoring definitions","",f"Recorded usage and estimated costs: {progress['totals']}","",
        "C = 0.35 F1* + 0.25 focal + 0.20 E + 0.20 B. S is the geometric mean of C across the four conditions. Missing required conditions or evidence classes remain unavailable; they are not silently reweighted. I is the focal expected/surprising × masked/unmasked interaction.","",
        "CSV exports include the usual 10,000-draw whole-run bootstrap intervals and Wilson focal intervals, conditional on these fixed datasets. Repeats are stochastic workflow replicates, not independent patient datasets.","",
        f"[Run metrics]({out/'run_metrics.csv'}) · [Condition metrics and intervals]({out/'completed_condition_metrics.csv'}) · [S and interaction]({out/'completed_cross_condition_scores.csv'}) · [Live progress]({root/'LIVE_PROGRESS.md'})"]
    atomic_write_text(out/"REPORT.md","\n".join(lines)+"\n")
    atomic_write_json(out/"metrics.json",dict(generated_at=now(),runs=rows,**all_tables))
    atomic_write_json(out/"provenance.json",dict(frozen_manifest_sha256=sha(root/"frozen_manifest.json"),
        report_hashes={r['source']:r['report_sha256'] for r in rows if r['report_available']},
        scorer_sha256=sha(Path(scoring.__file__))))


def run(root,workers):
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans
    from onc_co_scientist.expected_surprising.experiment import run_cell
    from onc_co_scientist.expected_surprising.coordination import WorkflowInfrastructureError
    import onc_co_scientist.expected_surprising.federation as module
    if not Path(module.__file__).is_relative_to(root/"source"):
        raise ValueError("Run with the frozen source on PYTHONPATH")
    lock=(root/"driver.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(root/"frozen_manifest.json")
    for name,expected in manifest["hashes"].items():
        if sha(root/name)!=expected:raise ValueError(f"Frozen file changed: {name}")
    grid=read(root/"grid.json");verify_grid(grid,manifest["model"])
    specs={c:load_experiment_spec(root/c/"config.yaml") for c in ("named","masked")}
    plans={c:{p.run_id:p for p in build_run_plans(s)} for c,s in specs.items()}
    pending=list(grid);active={};finished=[];cache={}
    state=dict(status="running",started_at=now(),pid=os.getpid(),workers=workers)

    def work(row):
        c,rid=row['condition'],row['run_id'];folder=root/c/'runs'/rid
        for retry in range(5):
            try:
                return run_cell(specs[c],plans[c][rid],root/c,specs[c].fingerprint(),resume=True)
            except WorkflowInfrastructureError as exc:
                # Archive only failed infrastructure calls; replay every successful scientific call.
                for path in (folder/'calls').rglob('*.json'):
                    record=read(path,{})
                    if record.get('result',{}).get('infrastructure_error'):
                        dest=folder/'infrastructure_archive'/f"retry-{time.time_ns()}"/path.relative_to(folder)
                        dest.parent.mkdir(parents=True,exist_ok=True);path.replace(dest)
                atomic_write_json(folder/'retry_status.json',dict(status='retry_wait',retry=retry+1,error=str(exc),updated_at=now()))
                if retry==4:raise
                time.sleep(min(60,10*2**retry))
        raise AssertionError('unreachable')

    def save():
        state.update(updated_at=now(),active=list(active.values()),queued_runs=len(pending),finished=finished)
        atomic_write_json(root/'execution.json',state)
        refresh(root,cache)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            while pending and len(active)<workers and not (root/'PAUSE').exists():
                row=pending.pop(0);active[pool.submit(work,row)]=row
            save()
            if not active:break
            done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                row=active.pop(future)
                try: result=future.result();status=result['status']
                except Exception as exc:
                    status='interrupted'
                    atomic_write_json(root/row['condition']/'runs'/row['run_id']/'worker_error.json',
                        dict(error=str(exc),traceback=traceback.format_exc(),updated_at=now()))
                finished.append(dict(condition=row['condition'],run_id=row['run_id'],status=status))
    state['status']='paused' if pending else 'scoring';save()
    if not pending:
        try:
            score(root);state['status']='finished'
        except Exception:
            state['status']='scoring_error';state['scoring_error']=traceback.format_exc()
        state['ended_at']=now();save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','refresh','score'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predecessor',type=Path)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--backend',choices=['gemini','gemma'],default='gemini')
    a=p.parse_args();root=a.root.resolve()
    if not 1<=a.workers<=12:raise ValueError('Use 1–12 concurrent experiments')
    if a.action=='prepare':prepare(root,a.predecessor.resolve(),a.backend)
    elif a.action=='run':run(root,a.workers)
    elif a.action=='score':score(root)
    else:refresh(root)


if __name__=='__main__':main()
