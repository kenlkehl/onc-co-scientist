"""Small local fixtures for resumable HF cache copies; no network or GPU needed."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("filelock")


def module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/expected_surprising/cache_caa_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("cache_caa_checkpoint", path)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.CHUNK = 2
    implementation.RANGE = 4
    return implementation


def test_copy_replaces_only_remote_link_and_preserves_partial_download(tmp_path):
    impl = module()
    data = b"0123456789abcdef"
    blob = hashlib.sha256(data).hexdigest()
    source = tmp_path / "remote/snapshots/revision"
    source.mkdir(parents=True)
    remote = tmp_path / "remote/blobs" / blob
    remote.parent.mkdir()
    remote.write_bytes(data)
    (source / "weights.safetensors").symlink_to("../../blobs/" + blob)
    target = tmp_path / "local"
    (target / "blobs").mkdir(parents=True)
    (target / "blobs" / blob).symlink_to(remote)
    seed = target / "blobs" / (blob + ".user.incomplete")
    seed.write_bytes(data[:3])
    report = tmp_path / "report.json"
    impl.copy_snapshot(source, target, report, workers=2, file_workers=2)
    cached = target / "blobs" / blob
    assert not cached.is_symlink() and cached.read_bytes() == data
    assert seed.read_bytes() == data[:3]
    assert remote.read_bytes() == data
    assert (target / "snapshots/revision/weights.safetensors").read_bytes() == data
    assert json.loads(report.read_text())["status"] == "completed"
    # Repeating the operation rehashes/reuses the verified local blob.
    impl.copy_snapshot(source, target, report)
    assert json.loads(report.read_text())["files"][0]["reused_local_blob"]


def test_range_resume_uses_bitmap_not_apparent_file_size(tmp_path):
    impl = module()
    data = b"0123456789abcdef"
    blob = hashlib.sha256(data).hexdigest()
    source, partial = tmp_path / "source", tmp_path / "copy.incomplete"
    source.write_bytes(data)
    partial.write_bytes(b"01234567xxxxcdef")
    journal = partial.with_suffix(partial.suffix + ".ranges.json")
    journal.write_text(json.dumps(dict(blob=blob, size=16, initial=4, complete=[0, 2])))
    impl.copy_ranges(source, partial, dict(blob=blob, bytes=16), workers=2, progress=lambda n: None)
    assert partial.read_bytes() == data
    assert impl.content_hash(partial, blob) == blob


def test_git_blob_metadata_hash(tmp_path):
    impl = module()
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    expected = hashlib.sha1(b"blob 2\0{}").hexdigest()
    assert impl.content_hash(path, expected) == expected
