"""Delete only the 120 explicitly authorized old Qwen/Gemma run directories."""
import json,shutil
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
old=repo/'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
new=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair'
ids=json.loads((new/'selection.json').read_text());assert len(ids)==120
snapshot=json.loads((repo/'outputs/clinical_workflow_restart_20260909/living_metrics.json').read_text())
(out/'pre_deletion_metrics.json').write_text(json.dumps(snapshot,indent=2))
rows=[]
for rid in ids:
 p=old/'runs'/rid
 assert p.parent==old/'runs' and not p.is_symlink()
 assert '__qwen_3_8_27b__' in rid or '__gemma_4_31b__' in rid
 d=json.loads((p/'run.json').read_text()) if (p/'run.json').exists() else {}
 rows.append({'run_id':rid,'status':d.get('status','interrupted'),'observed_usage':d.get('usage'), 'deleted_path':str(p)})
(out/'deleted_runs.json').write_text(json.dumps(rows,indent=2))
for rid in ids:shutil.rmtree(old/'runs'/rid)
assert not any((old/'runs'/rid).exists() for rid in ids)
print('Deleted 120 old vLLM run directories; Codex runs untouched',flush=True)
