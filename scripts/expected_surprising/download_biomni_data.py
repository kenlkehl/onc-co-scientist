"""Download the pinned Biomni data lake with atomic files and checksums."""

import argparse
import ast
import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download(root):
    source = root / "biomni/env_desc.py"
    tree = ast.parse(source.read_text())
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "data_lake_dict" for t in node.targets)
    )
    folder = root / "data/biomni_data/data_lake"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = root / "data/download_manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}

    def fetch(name):
        path = folder / name
        if path.exists() and name in previous and path.stat().st_size == previous[name]["bytes"]:
            return name, previous[name]
        url = "https://biomni-release.s3.amazonaws.com/data_lake/" + urllib.parse.quote(name)
        if path.exists():
            with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=120
            ) as r:
                if path.stat().st_size == int(r.headers["Content-Length"]):
                    with path.open("rb") as handle:
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    return name, {"url": url, "bytes": path.stat().st_size, "sha256": digest}
        temp = path.with_name(path.name + ".partial")
        digest = hashlib.sha256()
        offset = temp.stat().st_size if temp.exists() else 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            resumed = response.status == 206
            count = offset if resumed else 0
            if resumed:
                with temp.open("rb") as old:
                    while chunk := old.read(4 * 1024 * 1024):
                        digest.update(chunk)
            expected = int(response.headers.get("Content-Length", 0)) + count
            with temp.open("ab" if resumed else "wb") as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
        if expected and count != expected:
            raise ValueError(f"Incomplete download: {name}")
        temp.replace(path)
        print(f"Downloaded {name}: {count:,} bytes", flush=True)
        return name, {"url": url, "bytes": count, "sha256": digest.hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, name) for name in names]):
            name, record = future.result()
            previous[name] = record
            temp = manifest.with_suffix(".tmp")
            temp.write_text(json.dumps(previous, indent=2) + "\n")
            temp.replace(manifest)
    print(f"Verified {len(names)} Biomni resources", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    download(parser.parse_args().root.resolve())
