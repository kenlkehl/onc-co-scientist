import json

import pytest

from onc_co_scientist.harness.task_spec import (
    INSTRUCTIONS_FILENAME,
    SCHEMA_FILENAME,
    TASK_DATASET_LINK,
    TASK_DESCRIPTION_LINK,
    build_task,
    build_tasks,
)
from onc_co_scientist.synthetic.cancer_types import CancerType
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.io import write_bundle
from onc_co_scientist.synthetic.multi import (
    generate_multi_dataset,
    write_multi_bundle_pair,
)


def test_build_task_writes_agent_bundle(tmp_path):
    cfg = GeneratorConfig(
        dataset_id="ds_task",
        patient_n=60,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    dataset_dir = write_bundle(bundle, tmp_path / "ds")
    task = build_task(dataset_dir, tmp_path / "task", max_iterations=4)

    assert task.instructions_path.exists()
    assert task.schema_path.exists()
    assert task.example_path.exists()
    assert task.dataset_path.exists()
    assert task.description_path.exists()

    instructions = task.instructions_path.read_text()
    assert "ds_task" in instructions
    assert "Maximum iterations (N):** 4" in instructions

    # Agent-facing files must not leak the benchmark's evaluation intent.
    description_text = task.description_path.read_text().lower()
    instructions_lower = instructions.lower()
    for leak in (
        "open-mindedness",
        "deliberately inverted",
        "counter-intuitive",
        "paradigm",
        "willingness",
        "benchmark",
        "evaluat",
    ):
        assert leak not in instructions_lower, f"agent_instructions.md leaks: {leak!r}"
        assert leak not in description_text, f"dataset_description.md leaks: {leak!r}"

    # And the dataset description must not betray that the cohort is simulated.
    for tell in (
        "synthetic",
        "simulated",
        "benchmark",
        "evaluat",
        "data-generating",
        "ground truth",
        "generated for",
    ):
        assert tell not in description_text, f"dataset_description.md betrays synthetic nature: {tell!r}"

    # Schema file is valid JSON and includes the Transcript title.
    schema = json.loads(task.schema_path.read_text())
    assert schema.get("title") == "Transcript"

    # Critical: the ground-truth manifest is NOT shipped to the agent.
    assert not (task.task_dir / "manifest.json").exists()
    # And the dataset/description filenames referenced in the template exist.
    assert (task.task_dir / TASK_DATASET_LINK).exists()
    assert (task.task_dir / TASK_DESCRIPTION_LINK).exists()
    assert (task.task_dir / INSTRUCTIONS_FILENAME).exists()
    assert (task.task_dir / SCHEMA_FILENAME).exists()

    # Without --python-env, the brief contains no environment section.
    assert "Python environment" not in instructions


def test_build_task_embeds_python_env(tmp_path):
    cfg = GeneratorConfig(
        dataset_id="ds_env",
        patient_n=40,
        seed=0,
        n_concordant=1,
        n_discordant=0,
        n_hidden_novel=0,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    dataset_dir = write_bundle(bundle, tmp_path / "ds")

    env_dir = tmp_path / "agent_venv"
    env_dir.mkdir()

    task = build_task(
        dataset_dir, tmp_path / "task", max_iterations=2, python_env=env_dir
    )
    instructions = task.instructions_path.read_text()

    # The absolute path to the env should appear verbatim in the brief.
    assert str(env_dir.resolve()) in instructions
    assert "Python environment" in instructions


def test_build_tasks_mirrors_synth_tree(tmp_path):
    """build_tasks emits one task bundle per discovered named/anonymized
    bundle, mirroring the relative path under the output dir."""
    base = GeneratorConfig(
        dataset_id="ds_batch",
        patient_n=80,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
        n_extra_covariates=6,
    )
    chosen = [CancerType.crc, CancerType.breast]
    bundles = generate_multi_dataset(base, chosen)
    synth_root = tmp_path / "ds"
    write_multi_bundle_pair(bundles, synth_root, anon_seed=0)

    tasks_root = tmp_path / "tasks"
    tasks = build_tasks(synth_root, tasks_root, max_iterations=2)

    assert len(tasks) == len(chosen) * 2
    for ct in chosen:
        for variant in ("named", "anonymized"):
            task_dir = tasks_root / ct.value / variant
            instr = task_dir / INSTRUCTIONS_FILENAME
            assert instr.exists(), f"missing {instr}"
            text = instr.read_text()
            assert f"ds_batch_{ct.value}" in text
            assert "Maximum iterations (N):** 2" in text
            # Ground-truth manifest must not leak into the agent bundle.
            assert not (task_dir / "manifest.json").exists()
            assert (task_dir / TASK_DATASET_LINK).exists()
            assert (task_dir / TASK_DESCRIPTION_LINK).exists()
            assert (task_dir / SCHEMA_FILENAME).exists()


def test_build_tasks_raises_when_no_bundles_found(tmp_path):
    (tmp_path / "empty.txt").write_text("nothing here")
    with pytest.raises(ValueError):
        build_tasks(tmp_path, tmp_path / "tasks_out")
