# CRISPR dependency map `ds001_depmap`

This dataset contains 50000 cancer cell-line records from a CRISPR knockout dependency screen with CCLE-style molecular annotations. Cell-line feature names have been replaced with opaque labels (`feature_001`, `feature_002`, ...); dependency outcome columns retain their original gene names. More negative dependency scores indicate stronger dependency after knockout.

## Columns

### Identifiers
- `cell_line_id`

### Cell-line features
- `feature_011`
- `feature_010`
- `feature_004`
- `feature_005`
- `feature_001`
- `feature_026`
- `feature_016`
- `feature_025`
- `feature_019`
- `feature_015`
- `feature_014`
- `feature_009`
- `feature_022`
- `feature_024`
- `feature_029`
- `feature_035`
- `feature_020`
- `feature_027`
- `feature_012`
- `feature_018`
- `feature_008`
- `feature_006`
- `feature_032`
- `feature_021`
- `feature_030`
- `feature_028`
- `feature_007`
- `feature_031`
- `feature_013`
- `feature_033`
- `feature_003`
- `feature_036`
- `feature_023`
- `feature_034`
- `feature_002`
- `feature_017`

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
