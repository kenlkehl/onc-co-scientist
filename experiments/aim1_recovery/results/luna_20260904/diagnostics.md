# Post hoc recovery diagnostics

These checks describe the completed batch and do not change the primary outcome, thresholds, or exclusions.

| Family | Condition | N | Complete structure and membership match, before confirmation | Primary confirmed recovery | Strict confirmed recovery |
|---|---|---:|---:|---:|---:|
| clinical | named | 100 | 10 | 9 | 6 |
| clinical | anonymized | 100 | 5 | 5 | 5 |
| depmap | named | 25 | 0 | 0 | 0 |
| depmap | anonymized | 25 | 0 | 0 | 0 |

Incomplete structured definitions were the main limitation. No DepMap run submitted the full four-gate definition. Among submitted claims with the correct DepMap target, at most two defining subgroup predicates were matched or approximated. This is broader than a numerical-cutoff discrepancy. Many runs supported other or partial associations, which do not satisfy complete recovery.

In clinical runs, 10 named and 5 masked runs met structure and membership requirements before confirmation; 9 named and 5 masked runs met the full primary endpoint. The held-out evidence requirement therefore explains one lost recovery in this batch, not the overall low recovery rates.

This new pilot changes the model, structured output contract, neutral examples, and evaluation protocol. It does not isolate the effect of replacing the archived LLM judge, and it does not reproduce the archived clinical-versus-DepMap reversal. These results should not be inserted into the grant while retaining the old percentages or interpretation.
