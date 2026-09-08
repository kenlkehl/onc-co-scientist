# Full clinical comparison

80 assigned runs: 10 expected and 10 surprising per model, 25 iterations per run, on the same clinical dataset pair used in the smoke test.

| Model | Recall R | Precision P | F1* | Coverage E | Response to evidence B |
|---|---:|---:|---:|---:|---:|
| luna_medium | 0.053 | 0.320 | 7.438 | 17.267 | 71.644 |
| sol_medium | 0.642 | 0.630 | 61.579 | 62.844 | 75.198 |
| astra_medium | 0.675 | 0.841 | 74.598 | 65.433 | 63.756 |
| gemma_4_31b | 0.633 | 0.734 | 33.960 | 64.500 | 65.012 |

| Model | Expected focal recovery | Surprising focal recovery | Difference |
|---|---:|---:|---:|
| luna_medium | 0.000 | 0.000 | 0.000 |
| sol_medium | 1.000 | 0.900 | -0.100 |
| astra_medium | 1.000 | 1.000 | 0.000 |
| gemma_4_31b | 0.500 | 0.400 | -0.100 |

70/80 runs completed every stage without exhausted retries.

**Completion and model settings**

| Model | Runs finished | All stages successful | Successful stages | Reasoning / requested tier |
|---|---:|---:|---:|---|
| luna_medium | 20 | 20 | 2000 / 2000 | medium / priority |
| sol_medium | 20 | 20 | 2000 / 2000 | medium / priority |
| astra_medium | 20 | 20 | 2000 / 2000 | medium / priority |
| gemma_4_31b | 20 | 10 | 1979 / 2000 | Server settings / vLLM |

Stage failures and their existing score penalties are retained. Finished does not mean every stage succeeded. Codex tiers shown are requested settings; Luna, Sol, and Astra requested Priority, while Terra requests normal (`default`) tier.

Still pending; excluded from the score tables: qwen_3_8_27b: 7/20 runs finished; terra_medium: 0/20 runs finished.

Source batches: [luna_medium](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/manifest.json) · [sol_medium](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/manifest.json) · [astra_medium](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/manifest.json) · [gemma_4_31b](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_vllm/manifest.json)

Recall R is category-balanced sensitivity to embedded discoveries. Precision P is the fraction of final accepted claims that were tested and independently confirmed, including additional discoveries. R and P range from 0 to 1.

Within each run, F1* = 200 × R × P / (R + P), on a 0–100 scale. The asterisk marks category-balanced recall and confirmation-based precision, rather than one ordinary confusion matrix. An unrecovered stage error sets that run's primary F1* and focal recovery to zero under the existing failure rule; descriptive R and P are retained. The table averages run scores, so its F1* cannot be reconstructed from the displayed mean R and P, especially when runs incur this failure penalty.

**Coverage E: how broadly and how early did the agent test the embedded discoveries?** At the end of each iteration, calculate the fraction of expected, neutral, and surprising targets tested so far. Average those three fractions, then average across all 25 iterations:

**E = 100 × average over iterations of [(expected coverage + neutral coverage + surprising coverage) / 3].**

A target counts after a valid test of its exact comparison, regardless of the claimed direction or whether the agent accepts it. Repeated tests give no extra credit. Testing every target in iteration 1 gives E = 100; first testing them all in iteration 25 gives E = 4. The denominator stays at the configured iteration budget even if a run ends early, carrying its achieved coverage forward. Near matches are reported separately.

**Response to evidence B: how often did the agent's later decision agree with the evaluator's private interval rule?** For each independent validation result delivered in iteration t, score the assessment at synthesis in iteration t + 2. Agreement earns 1; disagreement or a missing due assessment earns 0. Both agent-requested and automatic validation count, including comparisons outside the embedded targets.

| Private evidence class | Interval rule | Decision that earns credit |
|---|---|---|
| Supported | Lower bound > private cutoff | Accept |
| Excluded | Upper bound < private cutoff | Reject |
| Ambiguous | Interval includes or touches private cutoff | Unresolved |

Intervals are oriented to the claim's direction. Here the private cutoff is 0.10 natural-log PFS units. Excluded means the interval rules out an effect at least that large; it does not necessarily rule out any association. Ambiguous means uncertain relative to that cutoff, not necessarily uncertain about the effect's sign.

**B = 100 × (supported agreement + excluded agreement + ambiguous agreement) / 3.**

Calculate each agreement fraction within a run, average available run fractions within each dataset version, then average versions equally (and base datasets equally when there is more than one). Finally average the three evidence classes equally. Thus B is not the fraction of all events correct or a simple average of run-level B scores. Runs without examples of a class do not contribute to that class's mean; if a class has no examples at the reporting level, B is unavailable. Invalid results, deadlines past the iteration budget, and checkpoints not reached because a run was interrupted are excluded. E and B both range from 0 to 100.

Agents judged effects without prescribed minima. Final confirmation used the private clinical cutoff of 0.10 natural-log PFS units. Agents did not need to request validation before accepting claims. The two-iteration checkpoint did not cap investigation.

Repeated runs were averaged within each version, then versions equally. B was averaged within evidence class before combining classes. This experiment contains only one base dataset, so confidence intervals over base datasets are unavailable.

[Paired summaries and denominators](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/full_clinical_comparison/paired_summaries.json) · [Run details](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/full_clinical_comparison/summary.json)
[Settings and source hashes](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/full_clinical_comparison/manifest.json) · [Execution record](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/full_clinical_comparison/execution.json)

Codex models used medium reasoning and five concurrent sessions per model. The CLI uses its default sampling and a 125,000 sampled-token rollout budget; these controls differ from the vLLM temperature and max-completion-token settings. Native CLI events and effective command arguments are archived under codex_calls.

**Why Astra scored lower on B**

Astra’s lower B mainly reflects how it used “accept,” “reject,” and “unresolved” relative to the evaluator’s undisclosed effect-size cutoff. The inspected narratives show it reading and discussing the validation results, including at the required checkpoint. This is not evidence that it generally ignored validation.

| Model | Supported: accept | Excluded: reject | Ambiguous: unresolved | B |
|---|---:|---:|---:|---:|
| Luna | 80.00% | 49.68% | 85.25% | 71.64 |
| Sol | 100.00% | 91.84% | 33.75% | 75.20 |
| Astra | 100.00% | 46.08% | 45.19% | 63.76 |

The percentages use the within-run and within-version averaging described above; they are not pooled event fractions.

**Compared with Sol:** Astra lost 15.25 B points on excluded evidence, gained 3.81 on ambiguous evidence, and tied on supported evidence: a net deficit of 11.44 points. Of Astra’s 80 excluded results, it rejected 36, left 42 unresolved, and accepted 2. All 42 unresolved cases had intervals crossing zero. Sol rejected 118 of its 129 excluded results. The contrast was especially pronounced for automatic validation: Astra rejected only 1 of 24 excluded results and left the other 23 unresolved.

For example, in Astra expected run 01, claim H48 tested the overall osimertinib–PFS association. Validation estimated 0.00394, with interval [−0.01081, 0.01869]. At iteration 14, Astra explicitly said this argued against a substantial association but did not establish either direction or exact equivalence; it retained “unresolved.” B required “reject” because the upper bound was below 0.10. Astra recognized the near-null result, but its label answered a different question. [Run 01 transcript](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/runs/astra_medium-nsclc_clinical-expected-r01/transcript.jsonl)

**Compared with Luna:** Astra gained 6.67 B points on supported evidence, lost 1.20 on excluded evidence, and lost 13.35 on ambiguous evidence: a net deficit of 7.89 points. Luna left 52 of 62 ambiguous results unresolved. Astra left 16 of 41 unresolved and accepted the other 25. Of those 25 accepted results, 23 had validation intervals entirely above zero, while still containing the private 0.10 cutoff.

For example, claim H73 in the same Astra run tested the BRCA2–PFS association. Validation estimated 0.0633, with interval [0.0198, 0.1069], close to the discovery estimate. At iteration 9, Astra described this as a modest, precise, reproducible descriptive association and retained acceptance, while declining a causal or treatment-predictive interpretation. B required “unresolved” because the interval crossed 0.10. Again, it used the evidence but did not apply the evaluator’s undisclosed magnitude standard. [Run 01 scored events](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/runs/astra_medium-nsclc_clinical-expected-r01/report.json)

**Interpretation:** B currently mixes response to evidence with agreement about a private minimum effect size. Agents were explicitly told to use their own judgment about effect sizes and conclusions. These examples therefore support a scoring/prompt mismatch as an explanation for Astra’s lower B, rather than a general failure of scientific reasoning. They do not establish that every disagreement was justified. Luna’s higher B also does not outweigh its much lower discovery recall and precision.

The models generated different comparisons, so their evidence pools are not identical. Eligible validation counts were 259 for Luna, 260 for Sol, and 258 for Astra; one additional Astra result arrived too late for its checkpoint and was excluded. Supported-class averages used 9 Luna runs versus all 20 Sol and Astra runs; the ambiguous-class average used 19 Astra runs. All other model/class averages used 20 runs. These differences and the single base dataset limit generalization. The scores have not been changed or rescored. [Audited counts and score components](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/B_DIAGNOSTIC.json)
