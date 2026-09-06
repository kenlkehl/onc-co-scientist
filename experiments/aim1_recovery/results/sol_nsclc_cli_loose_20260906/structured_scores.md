# Deterministic structured recovery

Version 2 primary recovery measures complete hypothesis identity at fixed membership tolerance; statistical confirmation is separate. Version 1 additionally gates primary recovery on evidence and the original contrast. Strict recovery requires equivalent boundaries under the selected version. Each JSON record identifies its version.

| Dataset | Condition | Primary | Strict | First primary iteration | Evidence |
|---|---|---:|---:|---:|---|
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 5 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 4 | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 3 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 6 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 3 | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 16 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 4 | heldout_20_percent |
| ds001_nsclc | named | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 5 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 1 | 1 | 6 | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 8 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 1 | 1 | 6 | heldout_20_percent |
| ds001_nsclc | named | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 4 | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 5 | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 5 | heldout_20_percent |
| ds001_nsclc | anonymized | 1 | 1 | 6 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 8 | heldout_20_percent |
| ds001_nsclc | named | 0 | 0 | censored | heldout_20_percent |
| ds001_nsclc | named | 1 | 1 | 4 | heldout_20_percent |
| ds001_nsclc | anonymized | 0 | 0 | censored | heldout_20_percent |

| Dataset | Condition | Confirmed | Interaction confirmed |
|---|---|---:|---:|
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | named | 1 | 1 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | named | 1 | 0 |
| ds001_nsclc | named | 1 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | named | 1 | 1 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | named | 1 | 1 |
| ds001_nsclc | named | 1 | 1 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | named | 1 | 0 |
| ds001_nsclc | named | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | named | 1 | 1 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 1 | 1 |
| ds001_nsclc | named | 1 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 1 | 0 |
| ds001_nsclc | named | 0 | 0 |
| ds001_nsclc | named | 0 | 0 |
| ds001_nsclc | named | 1 | 0 |
| ds001_nsclc | named | 1 | 0 |
| ds001_nsclc | named | 1 | 1 |
| ds001_nsclc | anonymized | 1 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | anonymized | 0 | 0 |
| ds001_nsclc | named | 0 | 0 |
| ds001_nsclc | named | 0 | 0 |
| ds001_nsclc | named | 1 | 0 |
| ds001_nsclc | named | 0 | 0 |
| ds001_nsclc | named | 1 | 1 |
| ds001_nsclc | anonymized | 0 | 0 |
