# Federated Azure cache transport and spending controls

The opt-in continuation uses the Azure Responses API directly. Existing `codex_cli` registry behavior, normal Codex authentication, single-site workflows, scientific controllers, prompts on disk, scoring, and frozen experiment inputs remain unchanged. Nothing in preparation starts an experiment or obtains credentials.

The transport changes are explicitly versioned as `azure-federation-cache-v1`. The new wire format preserves system/user/assistant message boundaries. It separates stable instructions, schema and aggregate task context from the changing ledger and peer drafts, places explicit cache breakpoints, and reuses a deterministic key for each model/research context. The iteration header moves to the end of its user message; every original character and every evidence reference is retained. The transport sends no tools, uses `store:false`, and makes no requests for patient rows. The server cache is distinct from the local completed-call replay journal.

Implicit caching is disabled so changing suffixes do not automatically incur cache-write charges. Persistent histories keep their original messages and earlier marked prefixes; the service limits new writes to its supported four breakpoints. Cache savings depend on actual Azure cache hits, history trimming, elapsed time, and how much context changes. Offline serialization tests do not establish a cache-hit rate.

The output ceiling remains 125,000 tokens. Existing per-deployment concurrency and throughput pacing are retained. Every HTTP attempt refreshes its Entra token after quota admission, with bounded retries for expired tokens, rate limits, and transport failures. There is no personal-account fallback or hidden SDK retry loop.

## Spending admission

A shared, file-locked ledger protects the whole batch across model processes. Before each attempt it reserves an upper estimate based on the full output ceiling, a conservative input byte bound plus framing allowance, and verified long-context rates. Successful usage settles the reservation using separate ordinary-input, cache-read, cache-write, and output rates. Reasoning tokens are already included in output. Amounts use integer microdollars with upward rounding.

Explicit HTTP 401/403/429 rejections release their reservations. Failed, disconnected, or interrupted requests without usage retain their full reservation; a restart never silently clears them. Thus retries cannot spend those reserved dollars twice. The default circuit breakers hold new work after three unknown-usage attempts or six consecutive misses for a previously observed reusable prefix. A hold does not turn an interrupted stage into a scientific failure. Completed results are still committed, and future dispatch stops.

A budget limits local request admission under the pinned rates and reservation bounds; it is not an Azure invoice or an Azure-side subscription cap. A reservation/usage mismatch holds further dispatch. Taxes, price changes and previously submitted requests with unknown billing remain outside an exact invoice guarantee. Rates expire after seven days and must match the saved Microsoft retail snapshot. Do not clear or recreate the ledger to resume; investigate outstanding reservations first.

## Prepare and inspect while held

The preparation script is restricted to the original 30 selected Sol/Terra/Luna identities. It verifies the original frozen manifest and driver locks, freezes an independent transport override under `control/cache_fix_v1`, and writes an **inactive $0 additional budget**. It refuses to overwrite an existing override or spending ledger.

```bash
PYTHONPATH=src /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/federation_cache_fix.py \
  --root /absolute/path/to/frozen/batch \
  --prepare --retail-rates /absolute/path/to/verified/azure_retail_snapshot.json
```

Inspect the prepared bundle without making any model or token-refresh calls:

```bash
PYTHONPATH=/absolute/path/to/frozen/batch/source/src \
  /tmp/ocs-es-refactor-venv/bin/python \
  /absolute/path/to/frozen/batch/control/cache_fix_v1/source/federation_cache_fix.py \
  --root /absolute/path/to/frozen/batch --check
```

Resume requires a separately authorized additional dollar cap, an enabled spending policy, explicit model releases, and validation of the new transport against a small, independently capped Azure canary. These are not enabled by implementation or preparation. Only then use the frozen override runner with `--resume-model sol_medium` (or Terra/Luna) and an explicit worker count. The runner reuses completed journals and returns completed runs without rerunning them. Do not launch the old CLI driver for the continuation.

The updated `monitor_federated_grid.py` reads small provider usage receipts instead of repeatedly loading large prompt journals. It shows input and output tokens, cache reads/writes, known costs, unknown accounting, paused state, and the additional spending ledger. Use the copy frozen with the override for this batch. Prior and future costs remain separately identifiable.

## Validation and sources

Offline tests cover lossless context splitting; stable and isolated cache keys; assistant-history preservation; shared spending admission; expiry/429 retries; unknown-cost retention; missing usage and cache-miss holds; cost bands; held preparation; completed-call replay; and scientific cancellation without a failed-call journal. Existing single-site, CLI, pacing and federation tests are run separately. No paid Azure cache validation is implied by these checks.

- [Azure prompt caching](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching)
- [OpenAI cache boundaries and message preservation](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Microsoft Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices)
