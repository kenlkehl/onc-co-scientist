# Federated context and decision history v2

The opt-in `federated_v2` policy addresses the repeated full-ledger history copies and short cacheable prefixes found in the initial Azure federation experiment. Existing single-site and legacy federation defaults retain their original coordinator and provider path. Default federation serialization omits the new policy field, preserving legacy configuration fingerprints. Normal Codex account configuration is untouched.

## Behavior

Persistent site teams and the central orchestrator retain complete prior agent responses with iteration/stage labels. Old user ledger snapshots are not repeatedly inserted into history; their originals remain in durable call journals. Every current request still receives the entire current ledger, notebook, response schema, and local aggregate context. History eviction removes oldest complete decisions, retains the newest complete response even if it exceeds the soft character target, and reports that overflow. Provider context admission and the shared spending cap remain hard bounds. This is an explicit memory-policy change, not lossless preservation of all historical prompts in model context.

For Azure, v2 factors the current task, claim definitions, and numerical evidence into an aggregate reference catalog before stage-varying developer instructions, history, mutable claim states, and peer drafts. Developer instructions keep their original priority and full text. Existing H/R references remain the only scientific reference IDs. Evidence is indexed by **H/R pair**, because a shared R result can have opposite signs for different hypotheses. Offline reconstruction must recover every original current-ledger value and evidence association exactly.

Catalogs and keys are isolated by run and site/task context, and reuse the same key across stages and peer/chair roles within that scope. Adding definitions/results appends records, preserving earlier cache boundaries. At most four explicit boundaries are marked. References omitted by the current request are excluded from the catalog rather than exposed as stale evidence. Inconsistent meanings for an existing H/R pair stop formatting rather than silently substituting data. Catalog ordering state is in memory only: after a process restart, a cold catalog is safe and expected; the catalog is not scientific state.

Azure request receipts include `context_layout.json` and transport version `azure-federation-context-v2`. The cache guard logs observed read/write amounts and reuse gaps, and counts warm misses only after eligible usage within the declared TTL. It retains the existing budget, unknown-usage protections, token refresh, bounded retries, output ceiling, and global pause behavior. These client changes do not fix or guarantee Azure retention/routing.

## Enable only in a separately versioned experiment bundle

Set:

```yaml
federation:
  site_counts: [2, 4]
  context_policy: federated_v2
```

Keep all other federation settings. For each Azure `codex_cli` model profile, supply a `budget_policy_path` pointing to the reviewed shared spending policy; the adapter derives an isolated namespace from the run's provider audit directory. A missing budget path fails before inference. The existing release/spending gates must still explicitly authorize model calls. Other provider types can use the federated decision-history policy through their existing adapters; the Azure catalog layout applies only to Azure.

Do not change the original paused experiment's frozen config/source or use v2 to resume its old prompt journals in place. Its history representation differs. Create a new bundle/version, retain old results for comparison, and declare the policy in the experiment fingerprint. One-site cells always use the legacy coordinator. Do not attach Azure-specific budget fields to model profiles intended for legacy single-site providers.

## Offline validation

The trace auditor performs no authentication or inference:

```bash
PYTHONPATH=src python scripts/expected_surprising/audit_federation_context_v2.py \
  --request-index /absolute/path/to/cache_investigation/request_audit.json \
  --output /absolute/path/to/context_v2/offline_trace_audit.json
```

On the 653 archived continuation requests, current-ledger reconstruction passed for every request. About 51.2% of formatted text is in the reusable catalog. Total formatted text was approximately 7.8% larger for these historical inputs, which retain whatever history was originally present. These are character counts, not token counts or measured savings. Coordinator tests separately check preservation of prior complete responses under large ledger pressure, accepted/rejected decisions, deterministic replay, and unchanged legacy/one-site behavior.

When cache savings are required for admission, a representative isolated canary should cover Sol, Terra, and Luna; 2/4 sites; persistent, sequential, and deliberative calls; stage/role changes; added and sign-oriented evidence; and 3/8/15-minute gaps. Compare cached input fraction and actual dollars with the original layout. Require complete usage receipts, exact ledger reconstruction, valid scientific forms, and useful savings on realistic full-size contexts; an immediate cache hit alone is insufficient. Charge any such diagnostic to the remaining authorized allowance. Keep the batch held on failure.

The prior six completed runs, 24 paused runs, existing cost ledger, and release gates are not changed by this implementation. No model calls are made by tests or trace auditing.

## Run without caching as an admission requirement

For an explicitly authorized uncached batch, set each Azure federation provider's
`prompt_caching: false` and its budget policy's `cache_miss_pause_enabled: false`.
The first setting removes cache breakpoints while retaining explicit mode, the full
scientific messages, and the output ceiling. The second makes cache observations
nonblocking. Both default to the previous behavior; single-site providers are unchanged.
Actual usage, reservations, missing-usage holds, price validity and spending caps remain
mandatory. Streamed admission rejections (`response.failed`, `rate_limit_exceeded`, no
usage and no output activity) use the bounded HTTP-429 retry/pacing path. Failures after
output activity retain their unknown-usage reservations.

`scripts/expected_surprising/launch_federation_science_v2.py` freezes a separately
versioned copy of the original initial-30 identities, datasets and committed code.
It transfers only the predecessor's remaining allowance into a new shared ledger,
leaving every old receipt/result and unknown reservation accounted for. Its observer
updates both the new bundle and the existing live-progress link, and closes release
and spending gates after those 30 finish or pause. It never dispatches the next batch.

Streamed `server_error`/`internal_server_error` failures with missing or invalid
usage now follow the bounded HTTP-500 retry path, including when inference began.
Their full unknown-cost reservations remain charged against admission. Retry-count,
unknown-attempt and spending limits still stop dispatch; completed responses with
invalid usage remain held rather than becoming scientific results.

For a transport-only continuation, drain drivers with the spending gate held,
then use `--launch --resume --transport-override PATH`. The override must match the
frozen provider's parsed source except for `_send`; its hash and the continuation
launcher's hash are recorded separately in `control/transport_override.json`.
Frozen source/configuration, per-run provenance, completed-call journals and the
shared spending ledger remain intact. Prior driver states are archived before
resume. Resumption neither transfers another allocation nor clears budget holds.
