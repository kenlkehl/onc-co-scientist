# Federated clinical workflow grid — 2026-09-12

2 and 4 random sites, named/masked presentations, expected/surprising datasets, three workflows, 10 repeats, 25 iterations. Prior single-site results supply the controls.

- [Frozen grid and operating instructions](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/README.md)
- [Full 1,680-run plan](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/grid.json)
- [Release policy](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/release_policy.json)
- [Source/input/config hashes](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/frozen_manifest.json)
- [Matched partition audit](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/partition_audit.json)
- [Launch record](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/launch.json)
- [Sol progress](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/control/sol_medium/STATUS.md)
- [Terra progress](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/control/terra_medium/STATUS.md)
- [Luna progress](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/control/luna_medium/STATUS.md)

Only Sol, Terra and Luna are released: 240 runs each, 10 concurrent runs per model. Astra, Claude, Gemma and Qwen are fully planned but held. No deferred model starts automatically.

Biomni's 80 native cells were initially [reserved](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/biomni_reserved_plan.json) and are now implemented in the separate native federation bundle, with external Qwen/xhigh orchestration and native Biomni at each site. They remain held. Live pilots also require a later explicit release. Qwen controller runs do not substitute for Biomni.

Validation: 30 federation/masking/experiment regression tests passed; both 840-run plans validated; all 720 released run identities are unique; site membership hashes match across all four data presentations for 2/4 sites and all 10 repeats; a held-model launch was rejected before creating any execution. New scripts pass Ruff.

[Live progress](/data1/ken/onc-co-scientist/data/expected_surprising_federation/20260912_clinical10pct_named_masked/LIVE_PROGRESS.md) refreshes every 30 seconds. [Native implementation and launch instructions](../../../docs/BIOMNI_FEDERATION.md).
