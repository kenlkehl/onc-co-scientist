# Masked six-model clinical workflow archive

Destination: `gs://kehl-lab-caia/onc-co-scientist-data/clinical-workflows/masked/six-model-grid/snapshots/20260911T010138Z`

Mirrors the unmasked upload's `runs/`, `reports/`, `provenance/`, and `metadata/` layout in a separate masked namespace. This is a live recursive copy; running jobs are not paused. Files may represent different moments during transfer. It is not an atomic or final snapshot. Upload success is recorded in metadata/completion.json.

- runs/: complete original directory names, frozen source/configuration, masked input data, private evaluation mappings, calls, native traces, reports and driver logs.
- reports/: captured masked live progress, interim metrics and matched masked/unmasked comparison (with their original snapshot dates).
- provenance/: restart, pause, deletion, validation and benchmark records.
- metadata/: selected_runs.json identifies the authoritative archived path for each of 360 assigned cells, including unfinished cells; manifest.json and completion.json describe the transfer.

The selected grid consists of 240 original masked Codex identities and 120 fresh masked Qwen/Gemma identities with recommended generation parameters. The retired masked vLLM repair root is included only for setup/provenance; its old run results were deleted by user request and must not be counted. Reports retain local paths; use selected_runs.json to resolve archive locations. Private evaluation maps are included to make the experiment reproducible and must not be exposed as agent input during benchmarking.

No unmasked run directories or Claude Opus campaign are included. Derived comparison reports include unmasked aggregate and matched metrics. Future snapshots should use clinical-workflows/masked/six-model-grid/snapshots/<UTC timestamp>/.
