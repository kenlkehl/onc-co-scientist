# Unmasked six-model clinical workflow archive

Destination: `gs://kehl-lab-caia/onc-co-scientist-data/clinical-workflows/unmasked/six-model-grid/snapshots/20260911T005709Z`

This is a live recursive copy begun September 11, 2026 UTC. Running jobs are not paused. Files can represent different moments during transfer; this is not an atomic or final snapshot. Upload completion is recorded separately.

## Folder layout

- `runs/`: full original directory names, including frozen configuration, source, inputs, call journals, native traces, per-run reports, and available driver logs.
- `reports/`: living report, machine-readable metrics, interim results, and workflow review as captured before upload.
- `provenance/`: restart audits, logs, diagnostics, benchmark setup, and deletion/cost records.
- `metadata/`: this guide, exact selected-run mapping, transfer manifest, and completion record.

The selected comparison has 360 cells: 136 retained original Codex runs, 104 repaired Codex runs, and 120 restarted Qwen/Gemma runs. `selected_runs.json` identifies the authoritative folder for each cell. Original superseded Codex traces are included for audit, not additional comparison replicates. Previously deleted Qwen/Gemma traces cannot be recovered; retained deletion/cost records are included.

## Stable namespace for later uploads

Use `clinical-workflows/<masking>/<campaign>/snapshots/<UTC timestamp>/`:

- `unmasked/six-model-grid/` (this archive)
- `masked/six-model-grid/` (future separate upload)
- `unmasked/claude-opus/` (future separate upload)
- `masked/claude-opus/` (future separate upload)

No masked or Claude Opus run directories are included here. Paths inside original reports/configurations are preserved and may refer to the original local checkout; use the relative selected-run mapping to find archived artifacts.
