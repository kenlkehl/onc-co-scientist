from pathlib import Path
import subprocess, json, datetime, shutil
r=Path('/data1/ken/onc-co-scientist'); out=Path(__file__).resolve().parent
m=json.loads((out/'manifest.json').read_text()); dest=m['destination']
reports=out/'reports'; reports.mkdir(exist_ok=True)
groups=[('runs',[str(r/'data/expected_surprising_ledger/full_runs'/name) for name in m['run_roots']]),('reports',[str(p) for p in reports.iterdir()]),('provenance',[str(r/p) for p in m['provenance']]),('metadata',[str(out/p) for p in ['README.md','selected_runs.json','manifest.json','upload.py']])]
for folder,sources in groups:
 print('Uploading '+folder,flush=True)
 subprocess.run(['gcloud','storage','cp','--recursive',*sources,dest+'/'+folder+'/'],check=True)
completed={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'destination':dest,'all_copy_commands_succeeded':True,'consistency':'Live non-atomic copy; not a final experiment snapshot.'}
(out/'completion.json').write_text(json.dumps(completed,indent=2)+'\n')
subprocess.run(['gcloud','storage','cp',str(out/'completion.json'),dest+'/metadata/completion.json'],check=True)
subprocess.run(['gcloud','storage','ls',dest+'/'],check=True)
print('UPLOAD COMPLETE '+dest,flush=True)
