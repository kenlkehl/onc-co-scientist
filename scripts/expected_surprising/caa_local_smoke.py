"""Bounded, offline BF16 derivation and two-arm generation smoke; not an efficacy test."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from onc_co_scientist.caa_server import CAAInferenceEngine, CaaModelAlias
from onc_co_scientist.interventions.caa import derive_caa_vectors, load_transformers_text_model
from onc_co_scientist.interventions.discovery_pairs import discovery_pairs, training_pairs
from onc_co_scientist.interventions.prompts import write_contrast_pairs


def main():
    # Leave useful thread diagnostics if cached checkpoint loading stalls.
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--stage-local", type=Path,
                        help="Copy the exact checkpoint to a new local temporary directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "started", "purpose": "engineering smoke, not scientific efficacy",
              "model": args.model, "dtype": "bfloat16"}

    def save():
        (args.out / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        import torch
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        model_path = args.model
        if args.stage_local:
            source = Path(args.model)
            files = sorted(p for p in source.iterdir() if p.is_file())
            size = sum(p.stat().st_size for p in files)
            if shutil.disk_usage(args.stage_local).free < size + 10 * 2**30:
                raise RuntimeError("Insufficient free disk for checkpoint staging plus 10 GiB")
            staged = Path(tempfile.mkdtemp(prefix="ocs-caa-gemma-", dir=args.stage_local))
            result["staged_checkpoint"] = str(staged)
            result["checkpoint_sha256"] = {}
            save()
            for file in files:
                print(f"Staging {file.name} ({file.stat().st_size / 1e9:.2f} GB)", flush=True)
                digest = hashlib.sha256()
                with file.open("rb") as src, (staged / file.name).open("wb") as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        dst.write(chunk)
                result["checkpoint_sha256"][file.name] = digest.hexdigest()
                if (file.is_symlink() and len(file.resolve().name) == 64
                        and digest.hexdigest() != file.resolve().name):
                    raise RuntimeError(f"Checkpoint content hash mismatch: {file.name}")
                save()
            model_path = str(staged)
        started = time.monotonic()
        processor, model = load_transformers_text_model(
            model_path, dtype="bfloat16", device_map="balanced", local_files_only=True,
        )
        result["load_seconds"] = time.monotonic() - started
        faulthandler.cancel_dump_traceback_later()
        result["device_map"] = {k: str(v) for k, v in model.hf_device_map.items()}
        if any(str(v) in {"cpu", "disk"} for v in model.hf_device_map.values()):
            raise RuntimeError("Model did not fit entirely on the GPUs")
        print(f"Model loaded in {result['load_seconds']:.1f}s", flush=True)
        pairs = training_pairs()
        write_contrast_pairs(pairs, args.out / "train.jsonl")
        write_contrast_pairs(discovery_pairs("development"), args.out / "development.jsonl")
        started = time.monotonic()
        bundle = derive_caa_vectors(
            pairs=pairs, processor=processor, model=model, layers=[20, 30, 40],
            position="last", enable_thinking=False,
            add_generation_prompt=False,
        ).with_random_control()
        bundle.metadata["model_id"] = model_path
        bundle.metadata["source_checkpoint"] = args.model
        bundle.metadata["checkpoint_sha256"] = result.get("checkpoint_sha256")
        bundle.metadata["validation_status"] = "unvalidated_constructed_appraisal_candidate"
        bundle.save(args.out / "vectors.npz")
        result["derivation_seconds"] = time.monotonic() - started
        print(f"26 contrast pairs derived in {result['derivation_seconds']:.1f}s", flush=True)
        aliases = [CaaModelAlias("gemma4-control", "control"),
                   CaaModelAlias("gemma4-caa", "candidate", scale=-0.05),
                   CaaModelAlias("gemma4-random", "random", scale=-0.05,
                                 concept="random_norm_matched")]
        (args.out / "aliases.json").write_text(json.dumps([asdict(a) for a in aliases], indent=2))
        engine = CAAInferenceEngine(
            model_path=model_path, vector_file=args.out / "vectors.npz", dtype="bfloat16",
            device_map="balanced", cache_implementation="dynamic", compact_agent_context=False,
            aliases={a.model_id: a for a in aliases},
        )
        engine.model, engine.processor, engine.vector_bundle = model, processor, bundle
        engine.manifest = engine._manifest()
        (args.out / "server_manifest.json").write_text(json.dumps(engine.manifest, indent=2))
        result["generations"] = []
        for alias, thinking in [(aliases[0], False), (aliases[1], False), (aliases[0], True)]:
            started = time.monotonic()
            generated, _, _ = engine.generate_for_request({
                "model": alias.model_id, "messages": [{"role": "user", "content":
                    'Return only the JSON object {"status":"ready"}.'}],
                "max_tokens": args.max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": thinking},
                "caa_expected_fingerprint": engine.manifest["fingerprint"],
            }, chat_messages=True)
            seconds = time.monotonic() - started
            result["generations"].append({"alias": alias.model_id, "thinking": thinking,
                "text": str(generated), "raw_text": generated.raw_text,
                "finish_reason": generated.finish_reason, "usage": generated.usage,
                "seconds": seconds, "tokens_per_second": generated.completion_tokens / seconds})
            print(f"{alias.model_id}: {generated.finish_reason}, {seconds:.1f}s", flush=True)
            save()
        result["peak_gpu_gib"] = [torch.cuda.max_memory_allocated(i) / 2**30
                                  for i in range(torch.cuda.device_count())]
        result["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
