# Clinical experiment API cost estimate — September 12, 2026

Current standard, global API list prices applied to the 540 Astra, Sol, Terra, Luna, and Opus runs selected in outputs/presentation_results/run_metrics.csv. These are API-equivalent token costs, not a reconstruction of subscription or Vertex invoices. All four expected/surprising × masked/unmasked conditions are included. Prior experiments, dataset generation, and report-writing costs are outside this cohort.

## Price assumptions

USD per million tokens. OpenAI rates: https://developers.openai.com/api/docs/pricing . Anthropic rates: https://platform.claude.com/docs/en/about-claude/pricing .

| Model | Input | Cached input | Cache write | Output |
|---|---:|---:|---:|---:|
| Astra | $10 | $1 | $12.5 | $50 |
| Sol | $4 | $0.4 | $5 | $20 |
| Terra | $2 | $0.2 | $2.5 | $12 |
| Luna | $0.2 | $0.02 | $0.25 | $1.2 |
| Opus 5 | $5 | $0.5 | $6.25 | $25 |

No fast, batch, flex, regional, or negotiated pricing adjustments. Cache writes were zero in the saved OpenAI usage. No recorded request exceeded the 272,000-input-token OpenAI long-context threshold. The Claude adapter does not request prompt caching; its input counts are priced as uncached. Claude input normalization merges cache categories, so this assumption must be revisited if its request configuration changes.

Formula: [(input − cached input − cache writes) × input rate + cached input × cache rate + cache writes × write rate + output × output rate] / 1,000,000. Output already includes reasoning/thinking tokens.

## Total recorded usage

Input includes the cached portion; do not add the cached column again. Totals include failed runs and saved retry usage.

| Model | Attempts / completed | Input, M | Cached input, M | Output, M | Estimated cost |
|---|---:|---:|---:|---:|---:|
| Astra | 120 / 119 | 678.958 | 56.088 | 16.787 | $7,124.15 |
| Sol | 120 / 118 | 809.863 | 53.562 | 36.332 | $3,773.27 |
| Terra | 120 / 118 | 505.532 | 36.866 | 11.180 | $1,078.86 |
| Luna | 120 / 113 | 591.306 | 1.918 | 15.988 | $137.10 |
| Opus 5 | 60 / 57 | 697.232 | 0.000 | 36.843 | $4,407.25 |

**Total: $16,520.64.** Pricing all recorded input as uncached instead gives $17,284.96.

## Mean recorded cost per completed run

Each workflow is averaged across all four conditions, weighted by its completed runs. Failed-run costs are included in totals above but excluded from these means.

| Model | Persistent | Sequential | Deliberative |
|---|---:|---:|---:|
| Astra | $57.23 | $30.93 | $89.99 |
| Sol | $23.56 | $17.88 | $52.82 |
| Terra | $10.19 | $4.53 | $12.23 |
| Luna | $1.09 | $0.63 | $1.71 |
| Opus 5 | $46.33 | $37.62 | $137.00 |

## Accounting limits

Known usage is reconstructed from each OpenAI provider_audit/call-*/metrics.json (cli_usage) and each Claude calls/*.json (result.usage). This recovers known input from runs whose aggregate input is null because of a missing call. Counts are never imputed for calls without usage. There are 126 stage calls marked as missing usage in the report and 140 unaccounted OpenAI infrastructure attempts; some of these may not have incurred any billable tokens. Unlogged usage is excluded, so totals can understate cost. One Astra audit includes an additional 388 output tokens beyond its report; the cost includes that saved provider invocation. No other output-count mismatches were found.

The per-run figures inherit the same missing-usage limitation, even for completed runs. Tool infrastructure, external compute, taxes, and platform surcharges are excluded.

## Reproduction

The adjacent estimate_clinical_api_costs.py contains the extraction and arithmetic; run it from the repository root with Python 3. It reads the saved report cohort and source audits, prints summaries, and writes /tmp/clinical_api_costs_runs.json. Prices are a frozen snapshot verified September 12, 2026; recheck the source pages before using this for future pricing.

## User-provided Vertex billing check

The user reports roughly $4,400 in the GCP console for completed Opus runs. This estimate is $4,107.34 for 57 completed runs and $299.90 for three failed runs, or $4,407.25 overall. The console amount is about 7.1% above the completed-only standard global API estimate. Invoice scope, missing usage, or Vertex-specific pricing could explain the difference; the logs alone do not resolve it. The rates and token-derived totals above are not retrofitted to the billing figure.
