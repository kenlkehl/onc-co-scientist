# External orchestration of native Biomni sites

The `external-native-federation-1` arm runs one Qwen/xhigh orchestrator outside
Biomni, with a native Qwen/xhigh Biomni scientist at each site. The orchestrator
owns a common claim ledger and directs scientific work; site workers retain the
native A1 research graph, resource retrieval, Python tools, conversation and
workspace. This is a distinct harness, with reporting rounds rather than the
controller's persistent/sequential/deliberative stage clock.

The September 12 grid contains 80 native cells: named/masked × expected/surprising
× 2/4 random sites × 10 repeats, with 25 reporting rounds. Its partitions use the
same pair/repeat/site-count seed rule as the other federated models.

## Round protocol

1. The external orchestrator sets a common goal and optional site-specific directions.
2. Each site conducts native research on its own partition, then submits a structured
   research handoff. Sites can recommend proposals, analyses, judgments and validation.
3. The orchestrator registers claims, selects official analyses, appraises evidence and
   optionally requests validation through the common benchmark gateway. It then calls
   `prepare_close`, which releases scheduled evidence and seals research submissions.
4. Sites review the shared ledger and their own canonical numerical results. Their
   native conversations and Python namespaces continue from the research dispatch.
5. The orchestrator uses these review handoffs to reassess new or due evidence and
   close the round. The next round continues from the same site workspaces.

Each site receives only its partition, public metadata and a private scratch mount.
Its exchange endpoint permits `state` and `handoff`; it cannot register official
claims, spend validation, or advance the round. Handoffs are typed recommendations,
not authoritative numerical receipts. The central model receives aggregate contexts,
proposals and trusted results, with no patient rows or site file access. Native
research text is not an automated disclosure guarantee; workers are instructed to
exclude records and identifiers, and arbitrary row/result fields are rejected.

Trusted code evaluates selected canonical comparisons in every site. Counts,
means and variances combine using the existing sufficient-statistics engine;
cells below 20 observations are suppressed and a required suppressed site makes
the combined result unavailable. The final independent evaluator is unchanged.

## Budgets, evidence and interpretation

One shared ledger allows 12 official analyses per reporting round and the existing
global voluntary/scheduled validation policy. Additional native exploratory code
is allowed within sites; it is not counted as another official benchmark test.
Expectations are recorded as not blinded to data inspection, as in the native
single-site arm.

The 4,200-request allowance covers central, site, resource-retrieval/helper and
failed requests. A concurrent reservation prevents sites from overspending the
shared allowance. All models use the pinned native Qwen/xhigh configuration and
adaptive output ceiling; no fallback model is substituted. Reports retain total
and per-site/central usage, unknown receipts, native completion rates, and scores.

## Preparation and launch gate

Preparation copies and hashes the native implementation, input packages, four
configurations and 80-cell plan. It makes no model calls and needs no Biomni
installation:

```bash
PYTHONPATH=src /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/biomni_federation.py prepare \
  --grid data/expected_surprising_federation/20260912_clinical10pct_named_masked
```

**Biomni remains held.** Both `pilot` and `run` check the parent
`release_policy.json` before preflight, dataset generation or model calls.
Only a subsequent explicit user instruction authorizes releasing `biomni_native`.
The launcher never changes this policy itself.

After release, use the frozen `biomni/source/biomni_federation.py` with
`biomni/source/src` on `PYTHONPATH` on the configured native worker host. Run
`pilot --grid <grid>` first. Each of the four configurations must pass both
expected/surprising full-length excluded pilots on freshly generated data.
Only then can `run --grid <grid>` execute the formal plan. Source, inputs, native
installation, environment and data-lake fingerprints must match the pilot gates.
The original native configuration points to `/home/kenneth_kehl/biomni`; that
installation and endpoint must pass preflight when launch is authorized.

## Replay and failures

Central responses and site handoffs have request/response hashes. A resumed
coordinator reconstructs ledger state using these records without rerunning
completed native dispatches. Invalid central JSON is also cached so replay does
not silently replace a failed response with a new draw. Repairs reuse site work.

Native continuation restores conversation, Python namespace, RNG state and
working directory inside the site sandbox. Checkpoint namespace hashes and
request/exchange counts must match. Unsafe checkpoints fail without replaying
arbitrary code. Completed and terminal failed runs remain final on `--resume`;
changed implementation/configuration requires a fresh frozen output root.

The live observer is read-only with respect to scientific runs and refreshes
`LIVE_PROGRESS.md` and `live_progress.json` in the parent grid. It observes
controller and native arms together; it never releases or starts a model.
