from pathlib import Path
import json,subprocess,datetime
repo=Path('/data1/ken/onc-co-scientist');out=Path(__file__).resolve().parent
m=json.loads((out/'manifest.json').read_text());dest=m['destination']
def run(cmd):
 print('COMMAND '+str(cmd),flush=True);subprocess.run(cmd,check=True)
try:
 run(['gcloud','storage','cp',str(out/'manifest.json'),dest+'/metadata/syncs/'+m['sync_id']+'/manifest.json'])
 for name in m['run_roots']:
  run(['gcloud','storage','rsync','--recursive',str(repo/'data/expected_surprising_ledger/full_runs'/name),dest+'/runs/'+name])
 run(['gcloud','storage','rsync','--recursive',str(out/'reports'),dest+'/reports'])
 for source in m['provenance']:
  run(['gcloud','storage','rsync','--recursive',str(repo/source),dest+'/provenance/'+Path(source).name])
 run(['gcloud','storage','rsync','--recursive',str(out/'metadata'),dest+'/metadata'])
 completion=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),destination=dest,all_sync_commands_succeeded=True,delete_remote_objects=False,sync_id=m['sync_id'],consistency=m['consistency'])
 (out/'completion.json').write_text(json.dumps(completion,indent=2))
 for target in ['metadata/syncs/'+m['sync_id']+'/completion.json','metadata/latest_sync_completion.json']:
  run(['gcloud','storage','cp',str(out/'completion.json'),dest+'/'+target])
 print('SYNC COMPLETE '+dest,flush=True)
except BaseException as exc:
 (out/'failure.json').write_text(json.dumps({'error':repr(exc),'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}));raise
