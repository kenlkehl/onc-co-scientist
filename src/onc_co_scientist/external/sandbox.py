"""Public-only mounts for a native code-executing external scientist."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def sandbox_command(spec, public, scratch, worker_config):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is required; no unsandboxed fallback")
    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-user",
        "--clearenv",
    ]
    for path in ("/usr", "/lib", "/lib64", "/bin", "/sbin"):
        if Path(path).is_symlink():
            command.extend(["--symlink", os.readlink(path), path])
        elif Path(path).exists():
            command.extend(["--ro-bind", path, path])
    # DNS/TLS only; do not expose arbitrary /etc credentials or host processes.
    for path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ssl", "/etc/fonts"):
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])
    environment = spec.python.parent.parent
    interpreter = spec.python.resolve().parent.parent
    mounts = [environment, interpreter, spec.biomni_root / "biomni"]
    if spec.python.is_symlink() and Path(os.readlink(spec.python)).is_absolute():
        mounts.append(Path(os.readlink(spec.python)).parent.parent)
    for path in dict.fromkeys(mounts):
        command.extend(["--ro-bind", str(path), str(path)])
    # Only the worker implementation is mounted; no repo or private evaluator.
    command.extend(
        [
            "--ro-bind",
            str(Path(__file__).with_name("biomni_worker.py")),
            "/worker.py",
            "--ro-bind",
            str(public),
            "/public",
            "--ro-bind",
            str(spec.data_lake),
            "/assets/biomni_data/data_lake",
            "--bind",
            str(scratch),
            "/work",
            "--ro-bind",
            str(worker_config),
            "/worker_config.json",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/assets/biomni_data/benchmark",
            "--dir",
            "/work/home",
            "--setenv",
            "HOME",
            "/work/home",
            "--setenv",
            "PATH",
            str(spec.python.parent) + ":/usr/bin:/bin",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "MPLBACKEND",
            "Agg",
            "--chdir",
            "/work",
            "--",
            str(spec.python),
            "/worker.py",
            "/worker_config.json",
        ]
    )
    return command
