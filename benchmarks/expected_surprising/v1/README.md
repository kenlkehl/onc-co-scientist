# Expected / surprising v1 historical release

**Withdrawn from evaluation.** The original literature review accepted unsupported population/comparator extrapolations and questionable neutral classifications. These files retain the historical DGPs and calibration audit. Current generation and evaluation require review policy 2.1.0, including independent candidate and full-DGP realism checks. Replacement data use `data/expected_surprising_v2/`.

10 paired datasets; 20 task instances. Each contains six embedded discoveries.

The specifications preserve literature review decisions and citation metadata. Full retrievals and model exchanges remain in `data/expected_surprising_v1/research/`. Expert adjudication is pending.

| Profile | Rows / version | Focal comparison | Mode | Calibration |
|---|---:|---|---|---|
| aml_clinical | 50,000 | aml_tp53_pfs | overall | Passed (1,000 replicates) |
| aml_depmap | 2,000 | aml_depmap_exp_kmt2a_men1 | subgroup | Passed (1,000 replicates) |
| breast_clinical | 50,000 | breast_clin_brain_mets_pfs | overall | Passed (1,000 replicates) |
| breast_depmap | 2,000 | depmap_erbb2_amplification_erbb2_dependency | subgroup | Passed (1,000 replicates) |
| crc_clinical | 50,000 | crc_pfs_bevacizumab_benefit | subgroup | Passed (1,000 replicates) |
| crc_depmap | 2,000 | crc_depmap_apc_ctnnb1_dependency | overall | Passed (1,000 replicates) |
| nsclc_clinical | 50,000 | nsclc_stage_iv_pfs | overall | Passed (1,000 replicates) |
| nsclc_depmap | 2,000 | cand_depmap_smarca4_smarca2 | subgroup | Passed (1,000 replicates) |
| prostate_clinical | 50,000 | prostate_psma_high_lu177_pfs | subgroup | Passed (1,000 replicates) |
| prostate_depmap | 2,000 | rb1_loss_e2f3_dep | overall | Passed (1,000 replicates) |

Replay the frozen specifications with:

```bash
ocs expected-surprising materialize benchmarks/expected_surprising/v1/specs data/expected_surprising_replay --historical-replay
```

Use the dependency versions in `release_manifest.json` for identical Parquet serialization. The source hashes record the generation implementation. Expose only the assigned public task to an agent.
