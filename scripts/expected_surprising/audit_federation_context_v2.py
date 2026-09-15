"""Offline replay of saved Azure requests through v2 layout; never calls a model."""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from onc_co_scientist.providers.base import ChatMessage
from onc_co_scientist.providers.federated_prompt import FederatedPromptLayout


def text(message):
    value = message["content"]
    return value if isinstance(value, str) else "".join(x["text"] for x in value)


def audit(index):
    rows = json.loads(index.read_text())
    paths = [Path(row["path"]).parent / "request.json" for row in rows]
    with ThreadPoolExecutor(max_workers=12) as pool:
        bodies = list(pool.map(lambda path: json.loads(path.read_text()), paths))
    layouts, totals, results = {}, Counter(), []
    for row, body in zip(rows, bodies, strict=True):
        # Every model/run has a distinct provider; the formatter further isolates sites.
        identity = (row["model"], row["run"])
        layout = layouts.setdefault(identity, FederatedPromptLayout("/".join(identity)))
        messages = [ChatMessage(m["role"], text(m)) for m in body["input"][2:]]
        content, key, meta = layout.render(messages, text(body["input"][1]), text(body["input"][0]))
        # This audit evaluates reference layout on the historical inputs exactly as recorded.
        # Decision-history retention is separately tested at coordinator level.
        old_size = sum(len(text(m)) for m in body["input"])
        size = sum(len(text(m)) for m in content)
        covered = sum(len(x["text"]) for x in content[1]["content"])
        result = dict(
            path=str(paths[len(results)]),
            **meta,
            original_characters=old_size,
            formatted_characters=size,
            catalog_fraction_characters=covered / size,
            cache_key=key,
        )
        results.append(result)
        totals.update(
            calls=1,
            original_characters=old_size,
            formatted_characters=size,
            catalog_characters=covered,
            ledger_roundtrips=1,
        )
    return dict(
        totals=dict(totals),
        requests=results,
        model_calls=0,
        note="Character proportions and layout parity; not measured Azure cache savings.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.request_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["totals"]))


if __name__ == "__main__":
    main()
