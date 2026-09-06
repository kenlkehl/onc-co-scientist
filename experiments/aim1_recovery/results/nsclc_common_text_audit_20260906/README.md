# DS001 NSCLC: common text recovery audit

The apparent Opus–Sol reversal is substantially amplified by a scoring mismatch.
Eight of the 17 masked Sol sessions scored as nonrecoveries already stated and
encoded the complete five-variable pattern, but used KRAS or sex as the exposure.
The older Opus scorer accepted analogous formulations. Under a common full-pattern
identity rubric, Sol is 14/20 named versus 11/20 masked, rather than 14/20 versus
3/20. Opus remains 8/20 named versus 17/20 masked. The observed direction still
reverses, but Sol's named advantage shrinks from 55 to 15 percentage points.

| Model and condition | Original reported recovery | Common text: explicit sotorasib contrast | Common text: complete joint pattern, any contrast orientation |
|---|---:|---:|---:|
| Opus 4.7 named | 8/20 (40%) | 8/20 (40%) | 8/20 (40%) |
| Opus 4.7 masked | 17/20 (85%) | 13/20 (65%) | 17/20 (85%) |
| Sol 5.6 medium named | 14/20 (70%) | 14/20 (70%) | 14/20 (70%) |
| Sol 5.6 medium masked | 3/20 (15%) | 3/20 (15%) | 11/20 (55%) |

All 80 saved research transcripts were included; no new scientist experiments
were run. These are retrospective text-identity judgments, with the same rules
applied to both models. They do not replace the original benchmark scores.

## What the masked Sol sessions actually found

The planted favorable cell is the conjunction:

```
sotorasib = 1
KRAS G12C = 1
ALK fusion = 0
BRCA2 mutation = 0
female sex = 0
```

For example, masked Sol job 0001, iteration 4, hypothesis `h10` states:

> `feature_016` equal to 1 increases `pfs_months` specifically in the complete subgroup `feature_018`=1, `feature_031`=0, `feature_028`=0, and `feature_005`=0.

The mapping is `feature_016` = KRAS G12C, `feature_018` = sotorasib,
`feature_031` = female sex, `feature_028` = ALK fusion, and `feature_005` = BRCA2
mutation. Its linked analysis reports a +4.923-month adjusted KRAS contrast in
that exact subgroup. Its structured finding faithfully records the same exposure
and all four conditions. This is a discovery of the complete favorable cell,
but its conditional comparison is KRAS-positive versus KRAS-negative among
sotorasib-treated patients. The production scorer requires sotorasib-treated
versus untreated within the target biomarker subgroup.

Seven extra masked Sol full-pattern recoveries use KRAS as exposure:
`0001`, `0009`, `0012`, `0017`, `0018`, `0033`, and `0034`. One, `0013`, uses
female sex as exposure, finding approximately 4.9 fewer PFS months for female
versus male patients with KRAS-positive, sotorasib-treated, ALK-negative,
BRCA2-negative disease. Its favorable cell is again the exact planted cell.

This is not simply a missing JSON field. The text and structured fields agree;
the chosen comparison differs from the treatment-specific scoring contract.
In this audit, every strict treatment-text positive in Sol was already a
production-score positive. The symbolic check also finds exactly the same 25
full-pattern positives across Sol's 40 sessions as the text review.

The 20 masked Sol sessions break down as follows:

| Category | Sessions | Count |
|---|---|---:|
| Complete pattern with the required sotorasib contrast | 0022, 0026, 0032 | 3 |
| Complete pattern with a different contrast orientation | 0001, 0009, 0012, 0013, 0017, 0018, 0033, 0034 | 8 |
| Partial planted pattern; missing one or both ALK/BRCA2 exclusions | 0004, 0016, 0019, 0024, 0040 | 5 |
| Search centered on masked pembrolizumab instead | 0003, 0007, 0021, 0025 | 4 |

## The matching issue also affects Opus

Four masked Opus sessions counted as recovered by the legacy scorer do not
explicitly isolate the exact sotorasib contrast in the saved full-pattern claim:
`run_001` and `run_002` use KRAS as exposure; `run_006` and `run_009` report the
joint combination/indicator. They receive full-pattern credit under the common
sensitivity rubric but not isolated-treatment credit. This accounts for the
17/20 versus 13/20 distinction in the table.

The older prose-matching prompt explicitly permits hypotheses proposing a
method capable of finding the generating process; the newer scorer requires
specific structured identity, exposure, direction, and subgroup. For the cases
identified here, the concrete discrepancy is chiefly contrast orientation.
The old recovery rate should not be compared directly to the new
exposure-specific rate as if both counted the same event.

## Reported analysis support

Identity and supporting analysis are separate checks. All 14 named and 11 masked
Sol full-pattern claims have linked reported support for their own comparison.
For Opus, all 17 masked full-pattern claims have reported support, while only
7 of the 8 named claims have a saved numerical analysis clearly testing the
complete conjunction. Named `run_016` states the complete subgroup as the
"cleanest responders," but its linked numerical results test broader subgroups
with one or both rare exclusions absent. It remains a text-identity positive
and is not credited with complete-joint reported support.

| Model | Named: complete pattern plus reported analysis | Masked: complete pattern plus reported analysis |
|---|---:|---:|
| Opus 4.7 | 7/20 | 17/20 |
| Sol 5.6 medium | 14/20 | 11/20 |

Reported support is based on the saved results and their links to hypotheses.
This audit does not independently rerun those analyses, certify all reported
numbers, or transfer held-out confirmation between different contrasts.

## Interpretation and limits

The benchmark currently mixes pattern recovery with identifying which opaque
column is the treatment. The masked dataset description lists predictors without
revealing their clinical roles, while the deterministic evaluator knows those
roles. The observed exposure swaps show that a session can recover the complete
interaction-defined cell and still fail the treatment-specific endpoint.
This makes the size of the original Sol naming gap a poor standalone measure of
failure to discover the underlying pattern.

A matched full-pattern comparison still shows a masked advantage for Opus and
a smaller named advantage for Sol. It does not establish a causal model effect:
these are 20 sessions per condition on one cohort, with different harnesses,
40,000 versus 50,000 discovery rows, recording contracts, stopping behavior,
and runtime settings. The audit was performed by the coordinating assistant,
unblinded to model and earlier results. The orientation-invariant sensitivity
was added after inspecting the masked Sol records; the joint-indicator extension
was added after inspecting Opus. It should be prespecified for a future experiment.

A useful future benchmark would report both full-pattern recovery and the
explicit treatment contrast. If the latter is the intended task, identifying
treatment columns consistently in both naming conditions would separate
clinical-role identification from subgroup discovery. No production scorer,
prompt, original transcript, or original result was modified by this audit.

## Evidence and reproduction

- [Protocol and retrospective sensitivity definition](protocol.md)
- [Per-session reviewer decisions](decisions.json)
- [Source-linked original quotations and linked analyses](evidence.json)
- [Sol symbolic favorable-cell cross-check](symbolic_crosscheck.json)
- [Comparison CSV](comparison.csv)
- [Source inventory and hashes](inventory.json)
- [Validation receipt](validation.json)

The audit script regenerates retrieval packets in Git-ignored
`data/nsclc_common_text_audit/` and rebuilds the tables and evidence from the
checked-in reviewer decisions:

```bash
python3 -m experiments.aim1_recovery.audit_nsclc_text
```

All 80 transcript hashes and all 79 available summary hashes were unchanged.
The archived masked Opus `run_010` has no summary file; its complete transcript
contains the required claim and linked analysis. Evidence retrieval used a broad
lexical screen, targeted review of hypotheses/results and summaries, and explicit
review of incomplete or ambiguous cases; lexical matches are not the classifier.
The independently computed Sol favorable-cell check preserves submitted predicates
and supplies no missing conditions from the answer key. It checks cell identity,
not equality of statistical estimands.
