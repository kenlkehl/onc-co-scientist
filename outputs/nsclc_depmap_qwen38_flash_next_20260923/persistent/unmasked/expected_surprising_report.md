# Aim 2 expected/surprising workflow results: nsclc_depmap_flash_next_20260923_persistent_unmasked

Each run used 25 iterations of explore, analyze, appraise, and synthesize. Each paired dataset version was run 10 time(s) per model and workflow. A repeat starts a separate agent conversation on the same dataset; it is not a new dataset.

The primary outcome is whether the workflow recovered the focal planted finding and the evaluator confirmed it independently. Scientific recovery is retained after stage failures; execution failures are reported separately. Expected and surprising percentages below show absolute performance. Their difference is surprising minus expected, in percentage points.

| Model | Workflow | Expected recovery % | Surprising recovery % | Difference (pp) | Failed / runs |
|---|---|---:|---:|---:|---:|
| qwen38_flash_next_xhigh | persistent | 0.0 | 0.0 | 0.0 | 1 / 20 |

The workflow comparison subtracts the persistent workflow's paired difference from each other workflow's paired difference. A positive value means the new workflow shifts relative recovery toward surprising findings. It does not by itself show higher overall accuracy; read it alongside both recovery percentages.

| Model | Workflow vs persistent | Change in paired difference (pp) | 95% interval (pp) |
|---|---|---:|---|

Results first average repeated runs within each dataset and version, then give each base dataset equal weight. Intervals resample whole base datasets, keeping their paired versions and workflows together. At least two base datasets in each included data type are needed for an interval; a one-dataset smoke test cannot estimate uncertainty across datasets.

The supporting outcomes describe the final shared research record:

- **R — recovery/recall:** the fraction of planted findings recovered, giving equal weight to expected, neutral, and surprising categories.
- **P — confirmed-claim fraction (precision/positive predictive value):** the fraction of final accepted claims that were tested during the run and confirmed in fresh evaluator data. Accepting a claim does not require prior validation. No accepted claims means P is unavailable.
- **F1\* — discovery performance:** the harmonic mean of R and P on a 0–100 scale. The asterisk distinguishes it from ordinary F1: R balances planted-finding categories, while P can include additional valid claims. Execution failures are reported separately; discoveries remain scored.
- **E — exploration coverage:** how broadly and how early the workflow tested planted findings across the full iteration budget, on a 0–100 scale.
- **B — evidence responsiveness:** balanced accuracy of responses to supportive, excluding, and ambiguous validation evidence, on a 0–100 scale. If a class never occurred, B is unavailable; this is missing evidence to assess, not a failure. The two-iteration response deadline is a reassessment checkpoint, not a run cap.

| Model | Workflow | R % | P % | F1* | E | B | Participant calls |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen38_flash_next_xhigh | persistent | 47.2 | 70.0 | 53.9 | 47.1 | 72.1 | 2153 |

The table averages each run's R, P, and F1* separately. Its F1* column therefore need not equal the harmonic mean of the displayed average R and P. Execution failure counts accompany the scientific scores.

All modes share analysis limits, validation samples, deadlines, final confirmation, and scientific scoring. Persistent mode retains recent committed conversation. Sequential mode uses a fresh conversation at each stage with the common ledger and notes. Deliberative peers see the same committed evidence and submit independent drafts; later rounds may see earlier peer drafts. The chair selects the only action that changes the research record. Drafts and disagreements remain in the call audit.

Persistent history is bounded at 120,000 characters of prior conversation. The complete current ledger and latest notebook are always provided. Oldest turns are removed only from active context, with each trim disclosed to the agent and recorded in the audit. This is not an iteration cap.

This comparison uses each workflow's natural number of participants. With two peers and one round, deliberation uses three calls per stage versus one in the other modes. It therefore compares complete workflows under equal scientific opportunity, not equal token or call spending. All attempts, including repairs and provider errors, are recorded. Missing provider token counts remain unavailable.

Machine-readable reports retain Q for P and D for F1* for compatibility. Each run includes the final scientific report, participant prompts and replies, and frozen input/code hashes. Secondary scores are unavailable for a whole condition if any assigned run lacks a scientific trace; the primary outcome still includes failures.

**Output-token use**

All participant responses count: linear agents, peers, the chair, and retries after invalid responses. Cached replay counts once. Provider output totals include reasoning where reported; reasoning subsets are not added again. Known totals are lower bounds when usage is missing. Unreported provider-internal work cannot be measured.

| Model | Workflow | Known output tokens | Mean per run | Missing call / run / transport counts |
|---|---|---:|---:|---|
| qwen38_flash_next_xhigh | persistent | 24,976,902 | 1248845.1 | 0 / 0 / 0 |

**How E and B are calculated**

E = 100 × the average over all iterations of mean exact-target coverage across expected, neutral, and surprising categories. A valid test counts regardless of acceptance or claimed direction; repeated tests add no coverage. Earlier testing earns more credit.

B = 100 × (supported agreement + excluded agreement + ambiguous agreement) / 3. The evaluator calls validation supported when its lower interval bound exceeds the private cutoff (credit for accept), excluded when its upper bound is below the cutoff (credit for reject), and ambiguous otherwise (credit for unresolved). Intervals are oriented to the claim. The assessment is scored two iterations after delivery. Class agreement fractions are calculated within runs, then averaged over repeats within version, versions, and datasets equally before combining classes. Missing due assessments score zero; invalid results and deadlines outside the reached or budgeted iterations are excluded. An absent class makes B unavailable.

| Model | Requested reasoning | Requested tier |
|---|---|---|
| qwen38_flash_next_xhigh | xhigh | unspecified |

vLLM accepts these fields, but model/template behavior determines their effect. Equal requested reasoning levels do not establish equal reasoning budgets.
