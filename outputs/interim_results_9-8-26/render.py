import collections
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path('/data1/ken/onc-co-scientist')
OUT=ROOT/'outputs/interim_results_9-8-26'
GRID=ROOT/'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
a=json.loads((OUT/'audit_results.json').read_text())
names={'luna_medium':'Luna','terra_medium':'Terra','sol_medium':'Sol','astra_medium':'Astra','qwen_3_8_27b':'Qwen 3.8 27B','gemma_4_31b':'Gemma 4 31B'}
def num(x,scale=1):
    return '—' if x is None else f'{x*scale:.1f}'
def focal(d,v):
    f=d.get('focal',{}).get(v,{'n':0,'recovered':0})
    return f"{f['recovered']}/{f['n']}" if f['n'] else '—'
def link(name,path):
    return f'[{name}]({path})'
when=datetime.datetime.fromisoformat(a['snapshot_at']).astimezone(ZoneInfo('America/New_York')).strftime('%B %d, %Y at %I:%M:%S %p EDT')
n=len(a['runs']); completed=sum(r['run']['status']=='completed' for r in a['runs'])
issues=[(r['run']['run_id'],x) for r in a['runs'] for x in r['issues']]
lines=['# Interim clinical workflow results — September 8, 2026','',f'**Snapshot: {when}.** {n} of 360 runs had finished: **{completed} completed all 100 scientific stages; {n-completed} finished with unrecovered errors.** Runs still active or queued are excluded, not scored as failures. This snapshot remains fixed while the experiment continues. All six models requested medium reasoning and standard/default service tier.','',
'**Preliminary grant data.** The frozen Codex adapter sometimes discarded a completed response after a reconnect warning. Retrying could change later decisions, and exhausted retries trigger score penalties. The current run continues unchanged at your request; the fix applies to subsequent experiments. Token totals below recover usage from saved native completion events, but scientific scores retain the original decisions and penalties.','',
'**What stands out:** Among all finished runs including failures, Astra persistent has the highest F1* (78.4), and Terra sequential has the highest B (99.0). Luna’s recovery is low despite frequent protocol completion. Qwen has only two fully successful runs so far. These are descriptive results from uneven, partially finished groups, not final rankings.','',
'**New interpretation issue:** at least 7 of Gemma’s 12 fully completed runs used an inflated clinical threshold in their final narrative: about 0.206 or 0.25 log-PFS units. Six explicitly computed 10% of mean log-PFS; the seventh used the same inflated 0.25 threshold. A 10% PFS increase is about 0.095 log units. This can make a reproducible effect of about 0.15 log units look clinically unimportant. Protocol completion therefore does not guarantee correct interpretation.','',
'## Runs that completed every stage','',
'These are the runs marked `completed`: 25 iterations of explore, analyze, appraise, and synthesize. Recovered retries are allowed. This table describes successful executions and can favor conditions with more failures; the next table includes those failures.','',
'R and P are percentages. F1*, E, and B use a 0–100 scale. Focal columns show recovered runs / observed runs for each dataset version. “Exp / Sur” gives the number of runs observed in each version. A dash means no runs or no eligible evidence, never zero.','',
'| Model | Workflow | Completed | Exp / Sur | Recall R % | Precision P % | F1* | Exploration E | Evidence response B | Focal expected | Focal surprising |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in a['conditions']:
    d=c['completed_only'];v=d.get('versions',{})
    lines.append(f"| {names[c['model']]} | {c['workflow']} | {d['n']} | {v.get('expected',0)} / {v.get('surprising',0)} | {num(d.get('R'),100)} | {num(d.get('P'),100)} | {num(d.get('F1'))} | {num(d.get('E'))} | {num(d.get('B'))} | {focal(d,'expected')} | {focal(d,'surprising')} |")
lines += ['', 'Qwen persistent has no B value in the successful-only table because its two successful runs had no eligible ambiguous-evidence events. Gemma deliberative has only expected-version completions so far; its successful-only scores describe that version alone.', '', '## All finished runs, including failures','',
'This is the more useful operational comparison. An unrecovered stage error makes F1* and focal recovery zero for that run, even when it found something useful. R, P, E, and B still describe the scientific record that exists. These are interim means, not the final balanced paired experiment.','',
'| Model | Workflow | Finished / 20 | Failed | Exp / Sur | Recall R % | Precision P % | F1* | Exploration E | Evidence response B | Focal expected | Focal surprising |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in a['conditions']:
    d=c['all_terminal'];v=d.get('versions',{})
    lines.append(f"| {names[c['model']]} | {c['workflow']} | {d['n']}/20 | {d.get('failed',0)} | {v.get('expected',0)} / {v.get('surprising',0)} | {num(d.get('R'),100)} | {num(d.get('P'),100)} | {num(d.get('F1'))} | {num(d.get('E'))} | {num(d.get('B'))} | {focal(d,'expected')} | {focal(d,'surprising')} |")
lines += ['', '## Output tokens for all finished runs','',
'Totals include peers, chairs, repairs, and provider retries. Means divide by the number of finished runs, including failures; they are ordinary per-run means. Native Codex completion usage replaces the coordinator total, so the same response is not counted twice. Reasoning tokens are already part of output tokens and are not added again. vLLM totals use the coordinator’s provider-reported usage.','',
'| Model | Workflow | Original observed | Reconciled observed | Mean / finished run | Recovered tokens | Original calls missing usage | Native attempts without completion |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for c in a['conditions']:
    d=c['all_terminal'];t=d.get('tokens',{});total=t.get('reconciled_observed',0)
    lines.append(f"| {names[c['model']]} | {c['workflow']} | {t.get('original_known',0):,} | {total:,} | {total/d['n']:,.0f} | {t.get('tokens_recovered',0):,} | {t.get('original_missing_calls',0)} | {t.get('native_attempts_without_completion',0)} |")
tot=collections.Counter()
for r in a['runs']:tot.update(r['tokens'])
lines += ['',f"Across these {n} finished runs: **{tot['reconciled_observed']:,} observed output tokens**, including **{tot['tokens_recovered']:,} recovered from native records** beyond the original {tot['original_known']:,}. {tot['native_attempts_without_completion']} native attempts have no completion usage; their unknown consumption is not imputed as zero. These totals exclude ongoing and queued runs and therefore differ from the live progress log.",'','## Pipeline audit','']
lines += [f"Checked all {n} terminal scientific reports and their saved call records. Frozen source, input, and configuration checksums: {len(a['hash_errors'])} mismatches across {a['frozen_files_checked']} files. Audit discrepancies: {len(issues)}.",'']
if issues:
    counts=collections.Counter(x for _,x in issues)
    lines += [f'- {count} × {label}' for label,count in counts.items()]
    lines += ['']
else:
    lines += ['The audit found no mismatches in report or call checksums, plan identities, dataset assignments, stage counts, call counts, or score calculations. All runs marked completed reached 25 iterations and committed each of the 100 stages exactly once. R/P/F1*, E, B, and focal recovery were recomputed from the saved scientific record using the frozen scoring code and matched the reports. This verifies scoring consistency; it does not rerun the models or regenerate independent validation datasets.','']
lines += [f"Codex native records contain **{tot['completed_with_warning']} completed attempts with error warnings** and **{tot['native_attempts_without_completion']} attempts without a completion event** ({tot['capacity_attempts']} explicitly report model capacity). The first category is the known adapter defect, not evidence that the server never produced an answer. Failures in the vLLM runs include malformed JSON and invalid stage forms or references; a finished run can preserve useful analyses while failing the protocol.",'',
'A review of final synthesis narratives found the inflated Gemma threshold in 3/6 completed persistent runs, 3/4 sequential runs, and 1/2 deliberative runs. This is a conservative count of explicit statements, not an exhaustive correctness score. For example, expected/persistent repeat 1 dismissed the marker-D effect of about 0.15 log units because it was below 0.206. '+link('Reviewed examples and exact run IDs',OUT/'threshold_interpretation.json')+'. For a future rerun, specifying that the 10% refers to PFS duration in months would clarify the intended meaning without prescribing a formula. The experiment and agent instructions were not changed during this audit.','',
'Gemma and Qwen scored zero on the ambiguous-evidence component of B in every observed workflow. This is not an absent evidence category: across their finished runs, Gemma had 18 eligible ambiguous events (14 reject, 3 accept, 1 missing due assessment), and Qwen had 3 (all reject); the reference response was unresolved. Qwen’s component is therefore based on very little evidence. '+link('Response counts',OUT/'ambiguous_evidence_responses.json')+'.','',
'Saved requests confirm the intended participant roles: single-agent calls in persistent/sequential workflows, and independent peers plus an authoritative chair in deliberative workflows. Peer drafts do not execute analyses. Persistent requests contained 2–12 messages as bounded history accumulated; sequential and deliberative requests each contained two messages, with deliberative drafts supplied to the chair in its system message. The audit checks Codex command settings for medium reasoning and default service tier. vLLM configuration requests the same settings, but this does not establish equal effective reasoning budgets across providers.','',
'## Reading the metrics','',
'- **R — recovery/recall:** fraction of embedded findings that were accepted, tested, and independently confirmed, giving equal weight to expected, neutral, and surprising findings. This is category-balanced recall. Exact matches are used here.','- **P — confirmed-claim fraction / precision:** accepted, tested, independently confirmed claims divided by all accepted claims. Additional discoveries can count. An agent may accept a claim before independent validation; validation determines whether the evaluator credits it. P is unavailable when there are no accepted claims.','- **F1\\* — discovery performance:** each run gets `200 × R × P / (R + P)` using R/P as fractions; zero if both are zero or P is unavailable. An unrecovered stage error also forces it to zero. The asterisk marks category-balanced recall and this failure rule, which make it differ from ordinary F1. The table averages per-run scores, so it is not the harmonic mean of the displayed R and P.','- **E — exploration coverage:** at each of the 25 iterations, calculate the fraction of target comparisons tested validly in each category; average the three categories, then average all 25 iterations and multiply by 100. Early coverage earns more credit. Repeating a test adds no coverage; acceptance is unnecessary.','- **B — response to evidence:** orient each validation interval in the claim’s direction. If its lower bound exceeds the private minimum, the credited response is accept; if its upper bound is below the minimum, reject; if it spans the minimum, unresolved. Score the explicit assessment at synthesis two iterations after evidence release. Average accuracy separately within supported, excluded, and ambiguous evidence, then average the three categories and multiply by 100. Missing due assessments receive zero; invalid or too-late evidence is excluded. If a category is absent for the condition, B is unavailable rather than reweighted.','',
'The **two-iteration window is a response deadline for B**, not a two-iteration run cap. Every run has a 25-iteration budget. Public instructions say that a relative outcome difference of at least 10% is clinically significant. The private evaluator still uses 0.10 natural-log PFS units (about a 10.5% geometric-mean difference); these thresholds are close but not identical.','',
'Within each model/workflow, the tables first average available runs within each dataset version, then give the observed versions equal weight. B follows that procedure separately for each evidence category. Unavailable P values are omitted and their counts are retained in the audit JSON. If only one version is available, its value is shown descriptively. Unfinished runs and missing partners are never invented or given zero scores. The final paired analysis requires all assigned repeats; these interim tables do not estimate that final contrast. There is only one underlying clinical dataset pair, so no across-dataset confidence intervals or model ranking claims are warranted.','',
'Persistent retains a bounded history of committed conversation (120,000 characters) plus the full current ledger and notes. Sequential starts each stage with a fresh conversation plus that ledger and notes. Deliberative uses two independent peers and one chair per stage. All share the same scientific analysis and validation budgets, but deliberative uses about 300 participant calls per clean run versus 100 for the other workflows.','',
'## Reproducibility','',
f"{link('Fixed snapshot and run IDs',OUT/'snapshot.json')} · {link('Per-run audit, denominators, and token reconciliation',OUT/'audit_results.json')} · {link('Read-only audit script',OUT/'audit.py')} · {link('Report renderer',OUT/'render.py')}",'',
f"{link('Frozen experiment configuration',GRID/'config.yaml')} · {link('Known adapter issue',GRID/'PROVIDER_ADAPTER_ISSUE.md')} · {link('Live progress',GRID/'LIVE_PROGRESS.md')}",'']
(ROOT/'interim_results_9-8-26.md').write_text('\n'.join(lines))
print(ROOT/'interim_results_9-8-26.md')
