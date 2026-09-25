# Local CAA integration checks

## 2026-09-14: local-cache smoke passed with CPU-staged transfers

The exact checkpoint was copied from SSHFS into genuine local ext4 storage:

`/home/klkehl/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343`

All seven files passed content-address verification, including full SHA-256 checks
of both weight shards (62,546,338,248 bytes total). The cache's previous remote blob
symlinks were atomically replaced by verified regular local files. Source weights,
HF refs and unrelated partial downloads were preserved. Receipt:
`outputs/caa_discovery/20260914_hf_cache_copy.json`.

The initial direct-transfer attempt loaded in 17.2 seconds and fit both GPUs, but
unsteered derivation produced non-finite values and was stopped without saving
vectors. Diagnostics reproduced failures with both SDPA and eager attention; rotary
sine/cosine inputs exceeded their expected [-1, 1] bounds. Small standalone
GPU-to-GPU copy checks passed, so these observations do not establish a particular
hardware, driver or library root cause. CPU-staged cross-GPU transfers made both
diagnostic prompts finite. The workaround is opt-in and model-scoped; it changes
neither weights nor installed packages/system settings. Its mode and implementation
are included in derivation metadata and the serving fingerprint.

The full rerun used BF16, balanced placement, SDPA, dynamic KV cache and
`--inter-gpu-transfer cpu`. It completed successfully:

| Check | Result |
| --- | --- |
| Load | 17.3 seconds; all weights on GPUs, no CPU/disk weight offload |
| Derive 26 contrast pairs at layers 20/30/40 | 16.1 seconds; finite candidate, orthogonalized and random-control vectors |
| Unsteered, thinking off | Correct JSON, normal stop; 6 tokens / 1.12 seconds |
| CAA layer 40, scale -0.05, thinking off | Correct JSON, normal stop; 6 tokens / 0.96 seconds |
| Unsteered, thinking on | Correct parsed JSON, normal stop; 82 tokens / 12.67 seconds including thought tokens |
| Peak allocated CUDA memory | 29.45 / 29.34 GiB on the two RTX A6000s |

The measured short-prompt generation rate was approximately 5.4–6.5 tokens/second,
including prefill and request overhead. This is not a long-context throughput or
maximum-context benchmark. Artifacts (vectors, manifest, corpora and detailed receipt):
`outputs/caa_discovery/20260914_local_hf_cpu_transfer_smoke/`.

The smoke now rejects non-finite activations/vectors and requires stopped, correct
JSON answers before marking success. Validation: 196 targeted CAA, transfer, cache-copy,
provider, controller, response-repair, masking, resilience and presentation tests
passed. Both GPUs returned to their initial idle allocations after the smoke.
No expected/surprising × named/masked discovery
campaign or efficacy evaluation was run. Constructed vectors remain scientifically
unvalidated; development gates and a short four-condition controller pilot are next.

## 2026-09-13 implementation check (historical)

### Implementation

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

### GPU and storage findings

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

### Next verification (as of September 13)

Stage the complete checkpoint on genuinely local storage using an authenticated,
efficient transfer or an explicitly budgeted SSHFS copy. Run `caa_local_smoke.py`
against that local snapshot. Only after successful loading, derivation, thinking
on/off generation, and a short four-condition controller pilot should a full matrix
be scheduled. Long-context KV-cache fit and aggregate throughput remain unmeasured.

The preliminary scientific design still needs independent development gates and
layer/scale selection; the implementation's synthetic corpus and default smoke scale
are not evidence that CAA reduces paradigm anchoring.
