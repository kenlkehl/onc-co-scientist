"""Fail-closed Linux filesystem isolation for model-authored Python.

The controller stays outside the sandbox. Only public inputs and a separate
analysis directory are mounted; receipts, other jobs and evaluator data are not.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path

from .runtime import run_subprocess_in_group

ISOLATION_VERSION = "bubblewrap-workspace-v1"
PUBLIC_INPUTS = (
    "dataset.parquet",
    "dataset_description.md",
    "metadata.json",
    "agent_instructions.md",
    "transcript_schema.json",
    "transcript_example.json",
)
ISOLATION_INSTRUCTIONS = (
    "Python runs in an isolated filesystem. Its working directory is /workspace. "
    "Read the supplied inputs there and save analysis scripts, outputs and "
    "analysis_summary.txt there using relative paths. Files you create persist between "
    "Python calls. Public inputs are read-only. Use sys.executable for child Python "
    "processes. Host paths, other jobs, repository source, evaluator data and network "
    "access are unavailable. The controller manages iteration records and transcripts "
    "outside your writable directory; use submit_iteration to submit records."
)


class SandboxUnavailable(RuntimeError):
    """Research must not start when the required isolation cannot be established."""


def analysis_root(workspace: Path) -> Path:
    marker = workspace / "filesystem_isolation.json"
    if not marker.exists():
        return workspace  # Existing CLI/archived workspaces retain their original layout.
    if marker.is_symlink() or json.loads(marker.read_text()).get("version") != ISOLATION_VERSION:
        raise ValueError("Invalid filesystem isolation marker")
    root = workspace / "analysis"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid isolated analysis directory")
    return root


class PythonSandbox:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.analysis = self.workspace / "analysis"
        if self.analysis.is_symlink():
            raise SandboxUnavailable("Analysis directory must not be a symlink")
        self.analysis.mkdir(exist_ok=True)
        self.bwrap = shutil.which("bwrap")
        if not self.bwrap:
            raise SandboxUnavailable("bubblewrap is required; unsandboxed execution is disabled")
        self.base = Path(sys.base_prefix).resolve()
        try:
            relative_python = Path(sys.executable).resolve().relative_to(self.base)
            self.python = Path("/runtime/python") / relative_python
        except ValueError as exc:
            raise SandboxUnavailable("Python executable is outside its runtime prefix") from exc
        self.sites = list(
            dict.fromkeys(Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib"))
        )
        self.verified = False

    def command(self, script: Path | None = None) -> list[str]:
        command = [
            self.bwrap,
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--assert-userns-disabled",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--hostname",
            "analysis",
        ]
        for name in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(name).exists():
                command += ["--ro-bind", name, name]
        for name in ("/etc/ld.so.cache", "/etc/fonts", "/etc/localtime"):
            if Path(name).exists():
                command += ["--ro-bind", name, name]
        command += ["--ro-bind", str(self.base), "/runtime/python"]
        sites = []
        for index, source in enumerate(self.sites):
            target = f"/runtime/site-packages/{index}"
            command += ["--ro-bind", str(source), target]
            sites.append(target)
        command += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/home",
            "--bind",
            str(self.analysis),
            "/workspace",
        ]
        for name in PUBLIC_INPUTS:
            source = self.workspace / name
            if source.is_symlink():
                raise SandboxUnavailable("Public input must not be a symlink: " + name)
            if source.exists():
                if not source.is_file():
                    raise SandboxUnavailable("Public input must be a regular file: " + name)
                command += ["--ro-bind", str(source), "/workspace/" + name]
        if script is not None:
            command += ["--ro-bind", str(script.resolve()), "/workspace/.agent_code.py"]
        for key, value in {
            "PATH": f"{self.python.parent}:/usr/bin:/bin",
            "HOME": "/tmp/home",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            # The parent uses -I -S; ordinary child Python needs these safe paths too.
            "PYTHONPATH": ":".join(sites + ["/workspace"]),
            **{
                key: os.environ[key]
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                if key in os.environ
            },
        }.items():
            command += ["--setenv", key, value]
        # -I -S excludes host PYTHONPATH, user site and executable .pth hooks.
        # Only the mounted package directories and this job's own helpers are added.
        bootstrap = f"import sys; sys.path[:0] = {sites + ['/workspace']!r}; "
        bootstrap += (
            "import runpy; runpy.run_path('/workspace/.agent_code.py', run_name='__main__')"
            if script
            else "print('sandbox-ready')"
        )
        command += ["--chdir", "/workspace", "--", str(self.python), "-I", "-S", "-c", bootstrap]
        return command

    def verify(self) -> None:
        if self.verified:
            return
        try:
            result = run_subprocess_in_group(
                self.command(),
                cwd=self.workspace,
                env={"PATH": os.defpath},
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable("Could not establish required Python isolation") from exc
        if result.returncode or result.stdout.strip() != "sandbox-ready":
            raise SandboxUnavailable("Required Python isolation failed: " + result.stderr[:1000])
        marker = self.workspace / "filesystem_isolation.json"
        if marker.is_symlink():
            raise SandboxUnavailable("Isolation marker must not be a symlink")
        marker.write_text(
            json.dumps(
                {
                    "version": ISOLATION_VERSION,
                    "guest_workspace": "/workspace",
                    "analysis_directory": "analysis",
                    "inputs_read_only": True,
                    "network": "disabled",
                    "controller_records": "not_mounted",
                },
                indent=2,
            )
            + "\n"
        )
        self.verified = True

    def run(self, script: Path, timeout: float):
        self.verify()
        return run_subprocess_in_group(
            self.command(script),
            cwd=self.workspace,
            env={"PATH": os.defpath},
            timeout=timeout,
        )

    def collect_summary(self) -> None:
        """Copy a regular summary only; never follow agent-created links on the host."""
        path = self.analysis / "analysis_summary.txt"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("Analysis summary must be a regular, unlinked file")
            content = source.read(1_000_001)
        if len(content) > 1_000_000:
            raise ValueError("Analysis summary exceeds size limit")
        with (self.workspace / "analysis_summary.txt").open("xb") as destination:
            destination.write(content)
