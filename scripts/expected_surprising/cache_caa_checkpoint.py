"""Copy an exact SSHFS HF snapshot into a genuinely local HF cache, resumably.

Existing blobs/snapshot entries are only replaced after content-address checks;
unrelated partial downloads and refs are never modified. No network client is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filelock import FileLock

CHUNK = 8 * 1024 * 1024
RANGE = 256 * 1024 * 1024


def copy_ranges(src, partial, entry, *, workers, progress):
    """Parallel, disjoint range reads; a durable bitmap makes holes resumable."""
    journal = partial.with_suffix(partial.suffix + ".ranges.json")
    if journal.exists():
        state = json.loads(journal.read_text())
        if state["blob"] != entry["blob"] or state["size"] != entry["bytes"]:
            raise RuntimeError("Copy-range journal does not match the source blob")
    else:
        initial = partial.stat().st_size if partial.exists() else 0
        if initial > entry["bytes"]:
            raise RuntimeError(f"Oversized partial blob: {partial}")
        state = dict(blob=entry["blob"], size=entry["bytes"], initial=initial, complete=[])
        journal.write_text(json.dumps(state))
    initial = state["initial"]
    ranges = [
        (start, min(start + RANGE, entry["bytes"]))
        for start in range(initial, entry["bytes"], RANGE)
    ]
    done = set(state["complete"])
    lock = threading.Lock()
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)

    def transfer(index):
        if index in done:
            return
        start, end = ranges[index]
        with src.open("rb", buffering=0) as reader:
            reader.seek(start)
            offset = start
            while offset < end:
                chunk = reader.read(min(CHUNK, end - offset))
                if not chunk:
                    raise RuntimeError(f"Unexpected EOF copying {src}")
                written = 0
                while written < len(chunk):
                    written += os.pwrite(fd, chunk[written:], offset + written)
                offset += len(chunk)
        os.fsync(fd)
        with lock:
            done.add(index)
            state["complete"] = sorted(done)
            temporary = journal.with_suffix(".tmp")
            temporary.write_text(json.dumps(state))
            temporary.replace(journal)
            progress(initial + sum(ranges[i][1] - ranges[i][0] for i in done))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(transfer, range(len(ranges))))
    finally:
        os.close(fd)
    return journal


def content_hash(path, expected):
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
    if len(expected) == 40:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def copy_snapshot(source, destination, report, *, workers=2, file_workers=8):
    source, destination, report = source.resolve(), destination.resolve(), report.resolve()
    entries = []
    for path in sorted(source.iterdir()):
        if not path.is_file() or not path.is_symlink():
            raise ValueError(f"Expected a content-addressed snapshot symlink: {path}")
        blob = path.resolve().name
        if len(blob) not in {40, 64} or any(c not in "0123456789abcdef" for c in blob):
            raise ValueError(f"Unrecognized content address: {path}")
        entries.append(dict(name=path.name, blob=blob, bytes=path.stat().st_size))
    destination.mkdir(parents=True, exist_ok=True)
    blobs = destination / "blobs"
    blobs.mkdir(exist_ok=True)
    snapshot = destination / "snapshots" / source.name
    snapshot.mkdir(parents=True, exist_ok=True)
    need = sum(
        e["bytes"]
        for e in entries
        if not (blobs / e["blob"]).is_file() or (blobs / e["blob"]).is_symlink()
    )
    if shutil.disk_usage(destination).free < need + 10 * 2**30:
        raise RuntimeError("Need checkpoint size plus 10 GiB of free local disk")
    state = dict(
        status="copying",
        source=str(source),
        destination=str(snapshot),
        total_bytes=sum(e["bytes"] for e in entries),
        files=entries,
    )
    state_lock = threading.Lock()
    report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with state_lock:
            partial = report.with_suffix(".json.tmp")
            partial.write_text(json.dumps(state, indent=2) + "\n")
            partial.replace(report)

    def copy(entry):
        src, target = source / entry["name"], blobs / entry["blob"]
        with FileLock(str(blobs / (entry["blob"] + ".caa-local-copy.lock"))):
            original_link = os.readlink(target) if target.is_symlink() else None
            if target.is_file() and not target.is_symlink():
                if (
                    target.stat().st_size != entry["bytes"]
                    or content_hash(target, entry["blob"]) != entry["blob"]
                ):
                    raise RuntimeError(f"Existing local blob is invalid; not overwriting: {target}")
                entry["reused_local_blob"] = True
            else:
                partial = blobs / (entry["blob"] + ".caa-local-copy.incomplete")
                if not partial.exists():
                    # Reuse a local prefix without modifying any pre-existing download.
                    seeds = sorted(
                        (
                            p
                            for p in blobs.glob(entry["blob"] + ".*.incomplete")
                            if p != partial
                            and not p.is_symlink()
                            and p.stat().st_size <= entry["bytes"]
                        ),
                        key=lambda p: p.stat().st_size,
                        reverse=True,
                    )
                    if seeds:
                        shutil.copyfile(seeds[0], partial)
                        entry["seeded_from"] = str(seeds[0])
                journal_path = partial.with_suffix(partial.suffix + ".ranges.json")
                done = (
                    json.loads(journal_path.read_text())["initial"]
                    if journal_path.exists()
                    else partial.stat().st_size
                    if partial.exists()
                    else 0
                )
                started = last_report = time.monotonic()
                initial = done
                print(
                    f"{entry['name']}: copying from {done / 1e9:.2f}/{entry['bytes'] / 1e9:.2f} GB",
                    flush=True,
                )

                def progress(copied):
                    nonlocal last_report
                    now = time.monotonic()
                    if now - last_report >= 15:
                        entry.update(
                            copied_bytes=copied,
                            copy_mbps=(copied - initial) / (now - started) / 1e6,
                        )
                        print(
                            f"{entry['name']}: {copied / 1e9:.2f}/{entry['bytes'] / 1e9:.2f} GB "
                            f"({entry['copy_mbps']:.1f} MB/s)",
                            flush=True,
                        )
                        save()
                        last_report = now

                journal = copy_ranges(src, partial, entry, workers=file_workers, progress=progress)
                entry["copied_bytes"] = partial.stat().st_size
                entry["status"] = "verifying"
                save()
                actual = content_hash(partial, entry["blob"])
                if partial.stat().st_size != entry["bytes"] or actual != entry["blob"]:
                    raise RuntimeError(
                        f"Content check failed for {partial}; retained for inspection"
                    )
                # Replaces a symlink itself; never opens the remote target for writing.
                partial.replace(target)
                journal.unlink()
                entry["previous_symlink"] = original_link
                entry["copy_seconds"] = time.monotonic() - started
            link = snapshot / entry["name"]
            wanted = "../../blobs/" + entry["blob"]
            if link.is_symlink():
                if os.readlink(link) != wanted:
                    raise RuntimeError(f"Conflicting existing snapshot symlink: {link}")
            elif link.exists():
                raise RuntimeError(f"Refusing to replace existing snapshot file: {link}")
            else:
                link.symlink_to(wanted)
            entry.update(status="verified_local", verified_content_hash=entry["blob"])
            print(f"{entry['name']}: verified local", flush=True)
            save()

    save()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(copy, entries))
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
    print(f"Local snapshot ready: {snapshot}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="Local HF cache repository directory (contains blobs/snapshots)",
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--file-workers", type=int, default=8)
    args = parser.parse_args()
    copy_snapshot(
        args.source,
        args.destination,
        args.report,
        workers=args.workers,
        file_workers=args.file_workers,
    )
