"""Read-only experiment observer: append progress without changing scientific artifacts."""
import collections
import datetime
import json
import pathlib
import sys
import time

root = pathlib.Path(sys.argv[1]).resolve()
plans = json.loads((root / 'plan.json').read_text())
seen = set()
returned = collections.Counter()
errors = collections.Counter()
tokens = collections.Counter()
missing = collections.Counter()
while True:
    groups = collections.defaultdict(collections.Counter)
    for plan in plans:
        key = (plan['model_profile'], plan['workflow_id'])
        directory = root / 'runs' / plan['run_id']
        state = 'running' if directory.exists() else 'queued'
        terminal = directory / 'run.json'
        if terminal.exists():
            try:
                run = json.loads(terminal.read_text())
                state = run.get('status', state)
            except (OSError, ValueError):
                pass
        groups[key][state] += 1
        for path in (directory / 'calls').glob('*.json'):
            if path in seen:
                continue
            try:
                result = json.loads(path.read_text())['result']
            except (OSError, ValueError, KeyError):
                continue
            seen.add(path)
            returned[key] += 1
            errors[key] += bool(result.get('error'))
            amount = result.get('usage', {}).get('output_tokens')
            missing[key] += amount is None
            tokens[key] += amount or 0
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    totals = sum(groups.values(), collections.Counter())
    summary = (f"{stamp} | runs queued={totals['queued']} active={totals['running']} "
               f"completed={totals['completed']} failed={totals['failed']} / {len(plans)} | "
               f"returned calls={sum(returned.values())} provider errors={sum(errors.values())} | "
               f"known output tokens={sum(tokens.values()):,} missing-usage calls={sum(missing.values())}")
    with (root / 'progress.log').open('a') as handle:
        handle.write(summary + '\n')
    lines = ['# Live clinical workflow progress', '', summary, '',
             'Refreshes every 30 seconds. Calls include peers, chairs, and retries; returned calls are not completed scientific stages. Token counts shown are provider-reported observed totals, including active runs; missing usage is flagged. Full final accounting is in the report.', '',
             '| Model | Workflow | Queued | Active | Completed | Failed | Returned calls | Provider errors | Known output tokens |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, counts in sorted(groups.items()):
        lines.append(f"| {key[0]} | {key[1]} | {counts['queued']} | {counts['running']} | {counts['completed']} | {counts['failed']} | {returned[key]} | {errors[key]} | {tokens[key]:,} |")
    temp = root / 'LIVE_PROGRESS.md.tmp'
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(root / 'LIVE_PROGRESS.md')
    execution = json.loads((root / 'execution.json').read_text())
    if execution.get('completed_at') or execution.get('failed_at'):
        break
    time.sleep(30)
