# Exploration and evidence responsiveness: 16-outcome persistent

| Version | Masking | E | B | Supported accuracy | Excluded accuracy | Ambiguous accuracy |
|---|---|---:|---:|---:|---:|---:|
| expected | unmasked | 44.44 | 71.41 | 100.00% | 90.22% | 23.99% |
| surprising | unmasked | 49.73 | 72.83 | 100.00% | 86.67% | 31.83% |
| expected | masked | 13.33 | 72.87 | 100.00% | 83.60% | 35.00% |
| surprising | masked | 5.93 | 73.16 | 100.00% | 86.16% | 33.33% |

Both E and B use a 0–100 scale.

Mean over 25 iterations of exact planted-finding test coverage, balanced across expected, neutral, and surprising categories, times 100. Earlier testing increases E. Acceptance is not required.

Average of supported, excluded, and ambiguous evidence-class response accuracies, times 100. Class accuracies are averaged across eligible repeats first, omitting repeats with no events in that class; classes receive equal weight. Delayed response is evaluated two iterations after evidence delivery. All assigned scientific traces, including the partially failed run, are retained.

B is conditional on evidence encountered; cells need not investigate the same claims. Eligible-run and event denominators are in the accompanying JSON.
