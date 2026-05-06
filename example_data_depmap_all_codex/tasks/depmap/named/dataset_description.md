# CRISPR dependency map `ds001_depmap`

This dataset contains 50000 cancer cell-line records from a CRISPR knockout dependency screen with CCLE-style molecular annotations. Columns include cell-line identifiers, lineage and molecular features, screen-quality covariates, and gene dependency scores. Dependency scores are centered so more negative values indicate stronger dependency after knockout, while values near zero indicate little selective effect.

## Columns

### Identifiers
- `cell_line_id`

### Cell-line features
- `lineage`
- `lineage_subtype`
- `culture_type`
- `screen_batch`
- `media_serum_pct`
- `cas9_activity_score`
- `growth_rate_doublings_per_day`
- `ploidy`
- `aneuploidy_score`
- `mutation_burden`
- `kras_mutation`
- `braf_v600e`
- `egfr_amplification`
- `erbb2_amplification`
- `pik3ca_mutation`
- `apc_mutation`
- `smad4_loss`
- `nf1_loss`
- `rb1_loss`
- `myc_amplification`
- `cdkn2a_loss`
- `pten_loss`
- `stk11_loss`
- `brca2_loss`
- `msi_high`
- `wnt_activity_score`
- `emt_score`
- `stemness_score`
- `ifn_gamma_signature`
- `hypoxia_score`
- `copy_number_8q24`
- `copy_number_9p21`
- `rnaseq_MYC_log2_tpm`
- `rnaseq_AXL_log2_tpm`
- `rnaseq_SLFN11_log2_tpm`
- `rnaseq_ASCL2_log2_tpm`

### Dependency outcomes
- `dependency_BRAF`
- `dependency_EGFR`
- `dependency_ERBB2`
- `dependency_KRAS`
- `dependency_RIT1`
- `dependency_KIF18A`
- `dependency_TMED10`
- `dependency_DHX9`
- `dependency_POLR2A`
- `dependency_RPL11`

Each row represents one cancer cell line; no missing values are present.
