# Common 25-iteration budget

Fresh expected/surprising tasks use 25 iterations for both clinical and cell-line (DepMap-style) datasets. No scientific or controller requirement for the earlier 25-versus-10 distinction was found. The shared budget gives both task types the same number of research stages and discovery-analysis opportunities.

The default applies to newly generated pairs and to packages copied from historical releases. Explicit historical replay and existing packages keep their recorded budgets. The original datasets, existing run directories, prompt forms, validation limits and schedules, and scoring formulas are unchanged. Six-iteration smoke overrides remain available. This update launched no live model runs.

Fresh packages are in `data/expected_surprising_ledger_25_iterations/`. All 20 tasks have a 25-iteration budget; every package-file hash and source numerical-dataset hash was verified. See [package audit](package_audit.json). Full clinical configuration templates and the [current guide](../../../docs/EXPECTED_SURPRISING_GUIDE.md) point to this root.

Validation releases remain at iterations 5, 12, and 20 for clinical tasks and 3, 6, and 8 for cell-line tasks, with reassessment two iterations later. The voluntary limit remains ten; the full-run comparison bound remains thirteen. Historical reports keep their original descriptions.

Verification: 76 tests passed across `test_expected_surprising.py`, `test_expected_surprising_workflow.py`, `test_expected_surprising_prompting.py`, and `test_expected_surprising_replicates.py`. This includes scripted full 100-stage runs for both task types, upgrading historical 10-iteration cell-line packages while preserving their source files and numerical bytes, and the existing smoke, prompt, and scoring tests. Ruff passed for the changed benchmark source, report generator, and test module.

The grant builder, both current grant DOCX files, and their text companion now say 25 iterations for both task families. Both DOCX files were rendered and all 15 pages inspected; the two changed statements were verified in the rendered PDFs.
