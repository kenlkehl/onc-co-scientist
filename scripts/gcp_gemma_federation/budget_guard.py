"""Stop the dedicated VM on controller exit/pause, with an optional budget deadline."""
import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def budget_status(deployment, now):
    start = datetime.fromisoformat(deployment['created_at'])
    stopped_at = deployment.get('cloud_stopped_at')
    accounting_end = min(now, datetime.fromisoformat(stopped_at)) if stopped_at else now
    enabled = deployment.get('budget_guard_enabled', True)
    return dict(updated_at=now.isoformat(), budget_guard_enabled=enabled,
        spending_cap_usd=deployment.get('budget_cap_usd') if enabled else None,
        conservative_runtime_estimate_usd=max(0, (accounting_end-start).total_seconds()/3600)
            * deployment['conservative_hourly_ceiling_usd'],
        basis='Conservative runtime envelope, not an actual GCP billing statement.',
        cloud_vm_status=deployment.get('cloud_status', 'running'),
        cloud_stopped_at=stopped_at,
        deadline=deployment.get('automatic_stop_deadline') if enabled else None,
        stop_on_controller_exit=True)


def pause_reason(root, status, now):
    if (root / 'PAUSE').exists():
        return 'experiment pause requested'
    deadline = status.get('deadline')
    if deadline and (datetime.fromisoformat(deadline)-now).total_seconds() <= 300:
        return 'budget deadline'
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    a=p.parse_args();root=a.root.resolve();d=json.loads(a.deployment.read_text())
    reason='controller exited'
    while True:
        now=datetime.now(UTC)
        d=json.loads(a.deployment.read_text())
        status=budget_status(d, now)
        write(root/'control/gcp-nvfp4/budget-status.json', status)
        try:
            cmd=Path(f'/proc/{a.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:
            break
        if str(root) not in cmd or 'resume_on_pool.py' not in cmd:
            break
        requested_pause=pause_reason(root, status, now)
        if requested_pause:
            reason=requested_pause
            (root/'PAUSE').write_text(reason+'; explicit release required before resume.\n')
            os.kill(a.pid,signal.SIGTERM)
            state=json.loads((root/'execution.json').read_text())
            active=state.pop('active',[])
            state.update(status='paused',active=[],interrupted_active=active,updated_at=now.isoformat(),pause_reason=reason)
            state.setdefault('finished',[]).extend(dict(condition=q['condition'],run_id=q['run_id'],status='paused') for q in active)
            write(root/'execution.json',state)
            break
        time.sleep(20)
    result=subprocess.run(['gcloud','compute','instances','stop',d['instance'],
        '--project='+d['project'],'--zone='+d['zone'],'--quiet'],capture_output=True,text=True,timeout=240)
    write(root/'control/gcp-nvfp4/guard-stop.json',dict(updated_at=datetime.now(UTC).isoformat(),
        reason=reason,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))


if __name__=='__main__':main()
