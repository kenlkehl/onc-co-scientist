# Azure-backed Codex experiments

The Codex CLI transport accepts `backend: azure` and an `azure_endpoint` ending in
`/openai/v1`. The provider uses the Responses API with a custom provider that does
not require OpenAI account authentication. It refreshes an Entra access token using
`az account get-access-token --resource=https://cognitiveservices.azure.com/`
before every CLI attempt. Transient token acquisition failures and rejected or expired
access tokens receive three retries with 2, 4 and 8 seconds of backoff by default
(`azure_auth_retries` and `azure_auth_retry_s`). An authentication rejection reruns
the exact same stage with another token acquisition and its own audited CLI attempt;
it does not consume a scientific-stage retry. Successful completions are not replayed.
Persistent authentication failures stop the request; there is no fallback
to a personal account. Credentials stay in subprocess memory and are not written
to config, command audit files, or exception messages.

Existing configurations retain their explicit ChatGPT transport by default.
For new Azure experiments, set these nonsecret provider fields in each Codex model
profile before freezing the bundle:

```yaml
provider_config:
  kind: codex_cli
  model_id: gpt-5.6-sol
  reasoning_effort: medium
  service_tier: default
  backend: azure
  azure_endpoint: https://YOUR-RESOURCE.openai.azure.com/openai/v1
```

This does not edit `~/.codex/config.toml`, change normal Codex sessions, or log the
user out. The configured model must exist at the endpoint under that deployment
name. Verify small, excluded calls before releasing formal runs.

## September 12 federation continuation

The user requested that the 30 active personal-account runs finish and that the
690 unopened Sol/Terra/Luna identities move to Azure. The original legacy drivers
cannot drain their queues. Each unopened output path was therefore reserved with
a small JSON **file**, before any provider could be initialized there. Existing
run directories were left untouched. The legacy `run_cell` rejects a reservation
with `NotADirectoryError`; these driver records are admission events, not scientific
failures or billed runs. Keep reservation files in place while retaining this bundle.

`azure_transition.json` in the original bundle records both disjoint selections.
`prepare_azure_federation.py` copies the original frozen scientific implementation
and data, changes only transport and admission code, preserves run identities,
and freezes the 690-identity continuation in a separate `_azure` bundle. The
combined progress observer follows those identities into the new bundle exactly
once. Held Astra, Claude, Biomni, Gemma and Qwen arms remain held.

`launch_azure_federation.py` is idempotent at the dispatch level. It waits for all
original drivers to have completed with no active or queued work, verifies every
original active identity has a terminal driver record and all reservations remain
intact, verifies the new bundle's hashes, checks fresh Azure credentials, and then
dispatches 10 workers per released model. The frozen scientific runner independently
enforces the same predecessor gate, so calling it directly cannot bypass the drain.

Run the launcher as a module from the repository, with its environment installed:

```bash
PYTHONPATH=src /path/to/python -m scripts.expected_surprising.launch_azure_federation \
  --root /path/to/federation_bundle_azure
```

While the predecessor is active it returns `waiting_for_original_runs` and starts
nothing. An automation can invoke this periodically and stay quiet until the state
changes. A launch receipt without a healthy execution record requires inspection;
the launcher intentionally does not auto-replay an ambiguous dispatch.

After switching, retain separate provider provenance and exclude all connection
checks from scientific results. Azure and personal-account usage must be reported
separately even when the named model and scientific configuration match.
