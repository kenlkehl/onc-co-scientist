import random

import pytest

from scripts.expected_surprising.prepare_federated_grid import balanced_selection
from scripts.expected_surprising.run_federated_grid import review_selection


def grid(model="sol"):
    return [
        dict(
            condition=c,
            semantic_condition=s,
            model_profile=model,
            workflow_id=w,
            site_count=n,
            partition_id="random",
            replicate=r,
            run_id=f"{model}-{w}-{n}-{r}-{c}-{s}",
        )
        for r in (1, 2)
        for w in ("persistent", "sequential", "deliberative")
        for n in (2, 4)
        for c in ("named", "masked")
        for s in ("expected", "surprising")
    ]


def test_review_blocks_match_all_conditions_and_models_independent_of_plan_order():
    orders = []
    for model, seed in (("sol", 1), ("terra", 2), ("luna", 3)):
        rows = grid(model)
        random.Random(seed).shuffle(rows)
        selection = balanced_selection(rows)
        assert len(selection) == len(rows)
        assert len({r["run_id"] for r in selection}) == len(rows)
        assert review_selection(selection, rows, 6) == selection[:24]
        lookup = {r["run_id"]: r for r in rows}
        orders.append(
            [
                tuple(
                    lookup[r["run_id"]][k]
                    for k in (
                        "workflow_id",
                        "site_count",
                        "replicate",
                        "condition",
                        "semantic_condition",
                    )
                )
                for r in selection
            ]
        )
    assert orders[0] == orders[1] == orders[2]


def test_incomplete_duplicate_or_mismatched_review_blocks_are_rejected():
    rows = grid()
    with pytest.raises(ValueError, match="Incomplete"):
        balanced_selection(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        balanced_selection(rows + [rows[0]])
    selected = balanced_selection(rows)
    for n in (0, -1, 100):
        with pytest.raises(ValueError, match="complete"):
            review_selection(selected, rows, n)
    with pytest.raises(ValueError, match="matched"):
        review_selection([selected[4], *selected[1:4]], rows, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        review_selection([selected[0], selected[0], *selected[2:4]], rows, 1)


def test_resume_cannot_silently_release_remaining_work():
    from scripts.expected_surprising.run_federated_grid import validate_review_resume

    validate_review_resume({}, None)  # Legacy non-review drivers retain their behavior.
    previous = dict(review_blocks=1, completed_at="finished", active=[])
    validate_review_resume(previous, 1)
    validate_review_resume(previous, 2)  # Explicit extension after review.
    for blocks in (None, 0):
        with pytest.raises(ValueError, match="review limit"):
            validate_review_resume(previous, blocks)
    with pytest.raises(ValueError, match="review limit"):
        validate_review_resume({**previous, "completed_at": None}, 2)
    with pytest.raises(ValueError, match="review limit"):
        validate_review_resume({**previous, "active": ["unfinished"]}, 2)
