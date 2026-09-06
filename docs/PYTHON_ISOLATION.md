# Structured runner filesystem isolation

The native `StructuredRunner` requires Linux bubblewrap (`bwrap`) and enabled
unprivileged user namespaces. It verifies isolation before its first model
request and fails if setup is unavailable. There is no unsandboxed fallback.
The model endpoint connection stays in the trusted controller; model-authored
Python has no network access.

Each Python call gets its own mount, user, PID and network namespaces. It sees
the assigned public inputs read-only at `/workspace`, plus writable analysis
files persisted in the host job's `analysis/` directory. Python and scientific
packages are mounted read-only under neutral runtime paths. Host home, dataset
repositories, sibling jobs, previous experiments, private evaluator data and
credentials are not mounted. Child processes inherit these restrictions;
creating nested user namespaces is disabled.

The controller's iteration records, submission receipts, logs, captured code and
final transcript live outside the writable analysis directory. The agent must
use `submit_iteration` to submit records. Fixed-budget script/output paths are
resolved within `analysis/`, with path escape checks. At completion the controller
copies a bounded regular `analysis_summary.txt` to the job root; links and special
files are rejected. `filesystem_isolation.json` records the layout/version.
Legacy CLI workspaces without that marker keep their existing artifact layout.

This isolation is applied by the shared native Python runner, including endpoint
runs with loose or fixed-budget prompts. It does not retrofit OS isolation into
independent Codex/Claude CLI runtime adapters. The scientific task brief, treatment
roles and structured hypothesis contract are unchanged.

Run the boundary and regression checks on a Linux host where bubblewrap can
create namespaces:

```sh
python -m pytest tests/test_python_sandbox.py tests/test_structured_runner.py \
  tests/test_research_budget.py tests/test_aim1_endpoint.py tests/test_aim1_local_cli.py
```

These checks use local fixtures, including six simultaneous sandbox workers, and
make no research-model calls. They deliberately fail if the host cannot enforce
isolation. An outer execution sandbox may deny namespace creation even when the
host supports it; run the tests with the host's normal runner permissions.

The invalid DS001 Qwen/vLLM batch and its setup runs from 2026-09-06 were deleted
after cross-workspace access was found. Their partial progress must not be resumed
or scored. Any replacement batch must be prepared anew with this isolation version.
