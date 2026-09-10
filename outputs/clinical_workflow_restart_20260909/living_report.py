"""Read-only living comparison over an explicit, versioned replacement selection."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo
from onc_co_scientist.expected_surprising.summary import _hierarchy
from onc_co_scientist.expected_surprising.scoring import EVIDENCE_CLASSES
from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text

REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OLD=REPO/'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
NEW=REPO/'data/expected_surprising_ledger/full_runs/20260909_codex_repair'
VLLM=REPO/'data/expected_surprising_ledger/full_runs/20260910_vllm_recommended'
vllm_ids=set(json.loads((VLLM/'selection.json').read_text()))
REPORT=REPO/'clinical_workflow_living_report_2026-09-09.md'
plans=json.loads((OLD/'plan.json').read_text())
selection=json.loads((OUT/'selection_audit.json').read_text())
selected={r['run_id'] for r in selection['rows'] if r['replace']}
names={'luna_medium':'Luna','terra_medium':'Terra','sol_medium':'Sol','astra_medium':'Astra','qwen_3_8_27b':'Qwen 3.8 27B','gemma_4_31b':'Gemma 4 31B'}
cache={}; call_cache={}

def read(path):return json.loads(path.read_text())
def number(x,scale=1):return '—' if x is None else f'{x*scale:.1f}'
def mean(values):
    values=[x for x in values if x is not None]
    return sum(values)/len(values) if values else None

def saved_report(folder):
    path=folder/'run.json'
    if not path.exists():return None,None
    st=path.stat();key=(str(path),st.st_mtime_ns,st.st_size)
    if key not in cache:
        result=read(path);rp=folder/'report.json';report=read(rp) if rp.exists() else None
        if report is not None and result.get('scientific_report_sha256'):
            assert hashlib.sha256(rp.read_bytes()).hexdigest()==result['scientific_report_sha256']
        cache[key]=(result,report)
    return cache[key]

def tokens(folder,report):
    if report and report.get('coordination',{}).get('output_token_accounting'):
        a=report['coordination']['output_token_accounting']
        return {'known':a['known_output_tokens'],'missing':a['missing_calls'],
                'unknown_internal_attempts':a['unaccounted_infrastructure_attempts']}
    known=missing=internal=0
    for p in (folder/'calls').glob('*.json'):
        if str(p) not in call_cache:
            r=read(p)['result'];u=r.get('usage',{})
            call_cache[str(p)]=(u.get('output_tokens'),max(0,(u.get('infrastructure_attempts') or 1)-1))
        n,k=call_cache[str(p)];known+=n or 0;missing+=n is None;internal+=k
    return {'known':known,'missing':missing,'unknown_internal_attempts':internal}

def update():
    groups=defaultdict(list)
    gemma_paused=VLLM.name=='20260910_vllm_repair' and (REPO/'outputs/qwen_generation_repair_20260910/gemma_pause.json').exists()
    qwen_paused=VLLM.name=='20260910_vllm_repair' and (REPO/'outputs/qwen_generation_repair_20260910/pause.json').exists()
    def inspect(p):
        rid=p['run_id'];root=VLLM if rid in vllm_ids else NEW if rid in selected else OLD;folder=root/'runs'/rid
        result,report=saved_report(folder)
        state=result['status'] if result else 'active' if folder.exists() else 'queued'
        if qwen_paused and p['model_profile']=='qwen_3_8_27b' and result is None:state='paused'
        if gemma_paused and p['model_profile']=='gemma_4_31b' and result is None:state='paused'
        return {'run_id':rid,'model':p['model_profile'],'workflow':p['workflow_id'],'version':p['semantic_condition'],'generation':'vllm_replacement' if rid in vllm_ids else 'replacement' if rid in selected else 'original','state':state,'folder':str(folder),'result':result,'report':report,'tokens':tokens(folder,report)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(inspect,plans))
    for item in records:groups[(item['model'],item['workflow'])].append(item)
    now=datetime.now(timezone.utc);totals=Counter(r['state'] for r in records)
    finished=totals['completed']+totals['failed'];stamp=now.astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %I:%M:%S %p EDT')
    replacement_finished=sum(r['generation']=='replacement' and r['result'] is not None for r in records)
    known=sum(r['tokens']['known'] for r in records);missing=sum(r['tokens']['missing'] for r in records)
    lines=['# Living clinical workflow report — September 9, 2026','',f'Updated **{stamp}**. The selected comparison contains **{finished}/360 finished runs** ({totals["completed"]} completed all stages; {totals["failed"]} finished with errors), **{totals["active"]} active**, and **{totals["queued"]} queued**, and **{totals["paused"]} paused**. **{replacement_finished}/104 Codex replacements have finished.**','',
    'This report follows the takeover of the six-model experiment. Each model has three workflows, two versions of the same clinical dataset, and ten separate runs per version: 20 runs per model/workflow. All models request medium reasoning and standard/default service. Each run has 25 iterations; the two-iteration rule is a deadline for reassessing evidence, not a limit on the run.', '',
    '## What changed','',
    ('**Qwen and Gemma batches are paused by user request.** Scheduled Codex monitoring is disabled. Generation settings are under review; saved results are preserved.' if gemma_paused else '**Qwen batch paused by user request.** Gemma continues unchanged; automatic Codex monitoring is disabled. Candidate Qwen sampling/JSON fixes are diagnostic only and have not been applied to this frozen batch.' if qwen_paused else ''),'',
    '- **104 Codex cells restart from the beginning:** 22 finished cells had native proof that the adapter discarded a completed response; 82 were still active or queued at the pause. Original traces remain saved. The 136 other finished Codex cells are retained, including four failures unrelated to this adapter defect. A replacement is chosen by its fixed run identity, never by whether its new score improves.',
    '- **Adapter:** a successful terminal completion after reconnect warnings is accepted, with its output tokens retained. Genuine terminal failures still fail. This fix is in the new frozen build; the old records are unchanged.',
    '- **Bookkeeping:** repeating a claim preserves its current scientific assessment. An unchanged assessment or investigation-only edit can proceed without evidence. These edits do not earn evidence-response credit, satisfy required reassessments, or change a scientific conclusion without evidence.',
    '- **Peer failure:** each peer retains its existing retry budget. If a peer still fails, the chair receives an explicit missing-draft notice and proceeds with available drafts and the ledger. The run records which stages used fewer peers. Chair output must still pass scientific checks; persistence or provenance errors still stop execution.',
    '- **Stage failure:** new runs retain discovery scores from their actual scientific record. Failed stages roll back, consume their attempts, and remain execution failures. The former whole-run zero penalty is retained as a separate diagnostic.',
    '- **vLLM restart September 10:** all 120 Qwen/Gemma runs start fresh with the repaired bookkeeping and failure policies. Old run directories were deleted at user request; aggregate cost/status records remain. Explicit reasoning boundaries are parsed before stage validation. Retry prompts include errors and rejected-response excerpts. Both models explicitly enable thinking with recommended sampling; only two consecutive truncations permit disabling it on the final retry. Normal calls request medium reasoning. Fallback requests are visible in call journals.', '',
    'The comparison still includes 136 retained original Codex runs and 104 repaired Codex runs. The new vLLM parsing/retry changes are separately versioned. Earlier Gemma threshold interpretation errors remain a scientific issue to assess, not a reason to change the hidden evaluator.', '',
    '## Progress by condition','',
    '| Model | Workflow | Retained original | Replacement | Queued | Active | Paused | All stages completed | Finished with errors |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    metrics=[]
    for (model,workflow),items in sorted(groups.items()):
        c=Counter(i['state'] for i in items);n=sum(i['generation']!='original' for i in items)
        lines.append(f'| {names[model]} | {workflow} | {20-n} | {n} | {c["queued"]} | {c["active"]} | {c["paused"]} | {c["completed"]} | {c["failed"]} |')
        terminal=[i for i in items if i['result'] is not None];rs=[i['report'] for i in terminal if i['report'] is not None]
        complete=len(rs)==len(terminal) and bool(rs)
        comp={cl:_hierarchy(rs,lambda r,cl=cl:r['responsiveness']['components'][cl]['accuracy']) for cl in EVIDENCE_CLASSES}
        b=None if any(v is None for v in comp.values()) else 100*sum(comp.values())/3
        d={'model':model,'workflow':workflow,'n':len(terminal),'failed':c['failed'],'versions':dict(Counter(i['version'] for i in terminal)),
           'R':_hierarchy(rs,lambda r:r['scores']['discovery']['exact']['R']) if complete else None,
           'P':_hierarchy(rs,lambda r:r['scores']['discovery']['exact']['Q']) if complete else None,
           'F1':_hierarchy(rs,lambda r:r['scores']['discovery']['exact']['diagnostic_D']) if complete else None,
           'F1_penalized':_hierarchy(rs,lambda r:0 if r['protocol_errors'] else r['scores']['discovery']['exact']['diagnostic_D']) if complete else None,
           'E':_hierarchy(rs,lambda r:r['scores']['E']) if complete else None,'B':b if complete else None,
           'degraded_runs':sum(bool(r.get('coordination',{}).get('degraded_stages')) for r in rs),
           'degraded_stages':sum(len(r.get('coordination',{}).get('degraded_stages',[])) for r in rs),
           'known_output_tokens':sum(i['tokens']['known'] for i in items),
           'missing_usage_calls':sum(i['tokens']['missing'] for i in items),
           'unknown_internal_attempts':sum(i['tokens']['unknown_internal_attempts'] for i in items),
           'mean_finished_output_tokens':mean([i['tokens']['known'] for i in terminal]),
           'B_components':comp}
        metrics.append(d)
    lines+=['','## Scientific performance so far','',
    'Includes finished runs with errors; active and queued runs are excluded. R and P are percentages. **Scientific F1\\*** uses actual accepted, tested, independently confirmed claims for every model, without the former whole-run failure penalty. **Penalty F1\\*** applies that old zero rule to every model for comparison. The saved original scores are not rewritten. Both columns describe the realized trace, not what an error-free rerun would have achieved.', '',
    'The two dataset versions receive equal weight among those available; runs within each version receive equal weight. A dash means insufficient observations or a missing evidence class. Uneven completion makes these interim comparisons provisional. The performance table combines retained and replacement runs; the preceding table identifies their counts.', '',
    '| Model | Workflow | Finished | Exp / Sur | Recall R % | Precision P % | Scientific F1* | Penalty F1* | Exploration E | Evidence response B | Runs with peer fallback |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in metrics:
        v=d['versions'];lines.append(f'| {names[d["model"]]} | {d["workflow"]} | {d["n"]} | {v.get("expected",0)} / {v.get("surprising",0)} | {number(d["R"],100)} | {number(d["P"],100)} | {number(d["F1"])} | {number(d["F1_penalized"])} | {number(d["E"])} | {number(d["B"])} | {d["degraded_runs"]} |')
    lines+=['','## Output tokens','',f'**{known:,} observed output tokens** across the selected comparison so far, including active runs. **{missing} returned calls lack output usage.** These are provider-reported totals; they are not a bill or a complete account of interrupted requests.', '',
    'Counts include peers, chairs, repairs, and returned retries. Reasoning tokens are already included in reported output tokens. No Codex native event totals are added to coordinator totals here, avoiding double counting. The final audit will replace coordinator totals with reconciled native completion totals where needed. Requests interrupted before a durable response and internal attempts without usage remain unknown.', '',
    '| Model | Workflow | Observed output tokens, active + finished | Mean observed / finished run | Returned calls missing usage | Internal attempts without separate usage |',
    '|---|---|---:|---:|---:|---:|']
    for d in metrics:lines.append(f'| {names[d["model"]]} | {d["workflow"]} | {d["known_output_tokens"]:,} | {number(d["mean_finished_output_tokens"])} | {d["missing_usage_calls"]} | {d["unknown_internal_attempts"]} |')
    def overhead(rid):
        folder=OLD/'runs'/rid;_,r=saved_report(folder)
        return tokens(folder,r)
    with ThreadPoolExecutor(max_workers=8) as pool:
        overhead_tokens=list(pool.map(overhead,selected))
    original_overhead=sum(t['known'] for t in overhead_tokens)
    overhead_missing=sum(t['missing'] for t in overhead_tokens)
    recovered=sum(e.get('output_tokens') or 0 for r in selection['rows'] if r['replace'] for e in r['adapter_evidence'])
    smoke=OUT/'live_smoke/results.json';smoke_tokens=sum(r.get('audit',{}).get('output_token_accounting',{}).get('known_output_tokens',0) for r in read(smoke)) if smoke.exists() else 0
    retirement=REPO/'outputs/vllm_restart_20260910'
    if (retirement/'pre_deletion_metrics.json').exists():
        prior=read(retirement/'pre_deletion_metrics.json')
        retired_tokens=sum(r['tokens']['known'] for r in prior['runs'] if r['run_id'] in vllm_ids)
        lines+=['',f'**Deleted vLLM run overhead:** {retired_tokens:,} observed output tokens in the last pre-deletion snapshot; this is a lower bound because in-flight calls may not have returned. Old vLLM traces were deleted by request. These tokens are excluded from the fresh comparison.']
    erased=REPO/'outputs/local_generation_restart_20260910/erased_runs.json'
    if erased.exists():
        prior=read(erased)
        lines+=['',f'**Additional erased unmasked-run overhead:** {sum(r["observed_output_tokens"] for r in prior):,} observed output tokens in the September 10 batch replaced by recommended-sampling runs. Excluded from current performance and token totals; interrupted unreturned calls remain unknown.']
    smoke_new=retirement/'live_smoke_retries/results.json'
    if smoke_new.exists():
        smoke_total=sum(r.get('audit',{}).get('output_token_accounting',{}).get('known_output_tokens',0) for r in read(smoke_new))
        lines+=['',f'Fresh vLLM retry-smoke checks used {smoke_total:,} observed output tokens, excluded from the comparison. Other preliminary smoke attempts and the thinking-toggle check are saved separately in outputs/vllm_restart_20260910.']
    lines+=['',f'**Separate restart overhead:** superseded original Codex traces contain {original_overhead:,} provider-recorded output tokens, plus {recovered:,} confirmed discarded-response tokens recovered by the selection audit. Their {overhead_missing} missing-usage call records overlap that recovered amount; full native reconciliation remains pending. Live prelaunch checks used {smoke_tokens:,} observed output tokens. These amounts are excluded from the selected comparison above, but preserved as experiment costs.', '',
    '## What the outcomes mean','',
    '- **R — recovery/recall:** fraction of embedded findings recovered, with equal weight to expected, neutral, and surprising findings. Recovery requires acceptance, a valid run-time test, and independent evaluator confirmation.',
    '- **P — confirmed-claim fraction / precision:** confirmed, tested accepted claims divided by all accepted claims. Agents can accept before validation; independent confirmation is part of scoring. Additional valid findings can count toward P.',
    '- **F1\\*:** harmonic mean of R and P, multiplied by 100. The asterisk distinguishes category-balanced recovery from ordinary F1. Scores are calculated per run before averaging; no accepted claims yields unavailable P and zero F1\\*.',
    '- **E — exploration coverage:** for each iteration, calculate the fraction of findings validly tested in each of the three finding categories. Average across categories and all 25 iterations, then multiply by 100. Early tests earn more credit; repeating a test adds none.',
    '- **B — evidence responsiveness:** score whether the explicit assessment at the response deadline agrees with supportive evidence (accept), excluding evidence (reject), or ambiguous evidence (unresolved). Missing due assessments score zero. Compute each evidence class\'s accuracy within each run, average runs and versions, then average the three classes and multiply by 100. If a class has no eligible observations, B is unavailable. Invalid evidence or deadlines the run never reached are excluded.',
    '- The public clinical threshold remains a 10% relative outcome difference. The unchanged private reference compares the claim-oriented validation interval with 0.10 log-PFS units, approximately a 10.5% geometric-mean difference. Lower bound above that reference supports acceptance; upper bound below it supports rejection; an interval spanning it calls for unresolved. This calculation is kept out of agent instructions.', '',
    'Persistent mode keeps bounded committed conversation plus the complete ledger and notes. Sequential mode starts a fresh conversation for each stage with the same ledger and notes. Deliberative mode ordinarily uses two independent drafts and one chair per stage; only the chair commits actions. It uses about three times as many participant calls for the same scientific action budget. Fallback stages are flagged above.', '',
    'These are preliminary results on one clinical dataset pair. They do not establish general model rankings or a causal benefit/harm of deliberation. Retained old-code runs and repaired-code runs must remain distinguishable in subsequent analyses.', '',
    '## Artifacts and progress logs','',
    f'- [Codex replacement progress log]({NEW}/control/progress.log)',
    f'- [Fresh local-model progress log]({VLLM}/control/progress.log)',
    f'- [Combined progress log]({OUT}/progress.log)',
    f'- [Replacement selection and native evidence]({OUT}/selection_audit.json)',
    f'- [Pause-time snapshot]({OUT}/snapshot.json)',
    f'- [Frozen replacement configuration]({NEW}/config.yaml)',
    f'- [Frozen replacement hashes]({NEW}/frozen_manifest.json)',
    f'- [Original interim report]({REPO}/interim_results_9-8-26.md)',
    f'- [Deliberative review]({REPO}/deliberative_workflow_review_2026-09-09.md)', '']
    payload={'at':now.isoformat(),'totals':dict(totals),'replacement_finished':replacement_finished,'conditions':metrics,'runs':[{k:v for k,v in r.items() if k not in {'report','result'}} for r in records],'known_output_tokens':known,'missing_usage_calls':missing}
    atomic_write_json(OUT/'living_metrics.json',payload)
    atomic_write_text(REPORT,'\n'.join(lines))
    with (OUT/'progress.log').open('a') as h:h.write(f'{now.isoformat()} finished={finished}/360 replacement_finished={replacement_finished}/104 active={totals["active"]} queued={totals["queued"]} observed_output_tokens={known:,}\n')
    return finished==360

if __name__=='__main__':
    while True:
        done=update()
        if '--once' in sys.argv or done:break
        time.sleep(60)
