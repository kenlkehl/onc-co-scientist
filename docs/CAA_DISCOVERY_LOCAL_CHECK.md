# Local CAA integration check — 2026-09-13

## Implementation

The new discovery path is described in `CAA_DISCOVERY.md`. It prepares a frozen,
jointly scheduled expected/surprising × unmasked/masked matrix with same-server BF16
controls, uses the existing scientific controller, and reuses the September metrics.
It does not reinterpret the historical Gemma gate as a valid experiment.

The generated candidate corpus is available locally under
`data/caa_discovery/constructed_appraisal_v1` (the repository ignores `data/`). The
corpus can be regenerated with the `pairs` command; it is unvalidated and constructed,
not new scientific results. No new Gemma vector or scientific-run result was produced
during this implementation check.

Validation: 177 targeted CAA, response-repair, discovery-controller, masking,
resilience, presentation-metric and provider tests passed. A host-side FastAPI HTTP
route smoke returned status 200 with exact token usage. The dependency lock passed
its offline consistency check. The actual September 10 named/masked input packages
also passed checksum, private-pair identity, and exact semantic-transformation checks.

## GPU and storage findings

The host exposes two NVIDIA RTX A6000 GPUs, each reporting 49,140 MiB total memory.
The cached BF16 checkpoint has 62,546,338,248 bytes across two safetensors shards.
Its snapshot is:

`/data1/ken/models/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343`

The apparent local `/data1` directory is actually the SSHFS mount
`klkehl@camus:/data1`. With both asynchronous and synchronous Transformers loading,
GPU allocations reached roughly 30 GB per device, but weight loading did not
complete. The synchronous loader was waiting on file pages during a tensor copy;
there was no observed CUDA out-of-memory exception. Allocations alone do not verify
successful model loading, maximum context, or inference throughput.

A temporary local staging copy was attempted with content hashing. It transferred
about 1.9 GB before being stopped because the SSHFS path was too slow for a bounded
smoke. Direct rsync from the mount's backing host was also tried, but SSH authentication
was unavailable in this session. The incomplete temporary copy was removed, original
weights were untouched, and both GPUs returned to their initial idle allocations.
Local attempt metadata remains in `outputs/caa_discovery/`.

## Next verification

Stage the complete checkpoint on genuinely local storage using an authenticated,
efficient transfer or an explicitly budgeted SSHFS copy. Run `caa_local_smoke.py`
against that local snapshot. Only after successful loading, derivation, thinking
on/off generation, and a short four-condition controller pilot should a full matrix
be scheduled. Long-context KV-cache fit and aggregate throughput remain unmeasured.

The preliminary scientific design still needs independent development gates and
layer/scale selection; the implementation's synthetic corpus and default smoke scale
are not evidence that CAA reduces paradigm anchoring.
