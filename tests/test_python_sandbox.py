"""Real namespace boundary tests; require Linux bubblewrap with user namespaces."""

import json
import os
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from onc_co_scientist.harness.python_sandbox import PythonSandbox, SandboxUnavailable
from onc_co_scientist.harness.structured_runner import StructuredRunner


def runner(workspace):
    workspace.mkdir(exist_ok=True)
    return StructuredRunner(workspace, base_url="http://unused", model="test")


def test_host_files_records_network_and_namespace_escape_are_blocked(tmp_path, monkeypatch):
    ws = tmp_path / "assigned"
    agent = runner(ws)
    secret = tmp_path / "other-job" / "analysis_summary.txt"
    secret.parent.mkdir()
    secret.write_text("other-job-canary")
    (ws / "private_evaluator.json").write_text("private-canary")
    (ws / "metadata.json").write_text('{"public": true}')
    protected = ("runner_transcript.jsonl", "submission_events.jsonl", "transcript.json")
    for name in protected:
        (ws / name).write_text("trusted-controller-record")
    monkeypatch.setenv("OPENAI_API_KEY", "credential-canary")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        code = f'''
import json, os, pathlib, socket, subprocess, sys
assert os.getcwd() == "/workspace"
assert json.load(open("metadata.json"))["public"]
assert "credential-canary" not in repr(dict(os.environ))
assert "{tmp_path}" not in os.environ.get("PYTHONPATH", "")
for path in {
            [
                str(secret),
                str(ws / "private_evaluator.json"),
                "../other-job/analysis_summary.txt",
                "/data1",
                "/home",
                "/proc/1/root" + str(secret),
                f"/proc/{os.getpid()}/root" + str(secret),
            ]!r
        }:
    try:
        with open(path) as f:
            f.read()
    except OSError:
        pass
    else:
        raise AssertionError("outside read succeeded: " + path)
for operation in [
    lambda: open({str(secret)!r}, "w"),
    lambda: os.chdir({str(secret.parent)!r}),
    lambda: os.listdir({str(tmp_path)!r}),
    lambda: open("metadata.json", "w"),
    lambda: os.unlink("metadata.json"),
    lambda: os.rename("metadata.json", "stolen.json"),
]:
    try:
        operation()
    except OSError:
        pass
    else:
        raise AssertionError("forbidden operation succeeded")
os.symlink({str(secret)!r}, "escape-link")
assert not pathlib.Path("escape-link").exists()
child = subprocess.run([sys.executable, "-c", {f"open({str(secret)!r}).read()"!r}],
                       capture_output=True)
assert child.returncode != 0
assert subprocess.run(["unshare", "-Ur", "true"], capture_output=True).returncode != 0
with socket.socket() as s:
    s.settimeout(1)
    assert s.connect_ex(("127.0.0.1", {listener.getsockname()[1]})) != 0
for name in {protected!r}:
    assert not pathlib.Path(name).exists()
    pathlib.Path(name).write_text("forged")
pathlib.Path("iterations").mkdir()
pathlib.Path("iterations/001.json").write_text("forged")
print("all boundaries held")
'''
        result = agent._python(code)
    assert result == "exit_code=0\nall boundaries held", result
    assert secret.read_text() == "other-job-canary"
    assert (ws / "metadata.json").read_text() == '{"public": true}'
    for name in protected:
        assert (ws / name).read_text() == "trusted-controller-record"
        assert (ws / "analysis" / name).read_text() == "forged"
    assert not (ws / "iterations").exists()


def test_scientific_python_child_imports_and_persistent_outputs(tmp_path):
    agent = runner(tmp_path)
    agent.python_timeout = 120  # Allow cold scientific-library imports on shared storage.
    result = agent._python("""
import numpy, pandas, scipy, statsmodels.api, sklearn, pyarrow
import subprocess, sys
subprocess.run([sys.executable, "-c", "import pandas; print('child imports work')"], check=True)
pandas.DataFrame({"x": [1, 2]}).to_parquet("result.parquet")
open("analysis_summary.txt", "w").write("Found an association.")
""")
    assert result == "exit_code=0\nchild imports work", result
    assert agent._python('import pandas; print(pandas.read_parquet("result.parquet").x.sum())') == (
        "exit_code=0\n3"
    )
    agent._sandbox().collect_summary()
    assert (tmp_path / "analysis_summary.txt").read_text() == "Found an association."


def test_six_concurrent_workspaces_are_independent(tmp_path):
    def work(index):
        agent = runner(tmp_path / str(index))
        result = agent._python(f'open("result.txt", "w").write({str(index)!r}); print("ok")')
        assert result == "exit_code=0\nok", result
        return (agent.workspace / "analysis/result.txt").read_text()

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(work, range(6))) == list(map(str, range(6)))


def test_background_child_cannot_survive_python_call(tmp_path):
    agent = runner(tmp_path)
    result = agent._python('''
import subprocess, sys
subprocess.Popen([sys.executable, "-c",
                  "import time; time.sleep(2); open('late-write', 'w').write('escaped')"],
                 start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("parent done")
''')
    assert result == "exit_code=0\nparent done", result
    result = agent._python('''
import pathlib, time
time.sleep(3)
assert not pathlib.Path("late-write").exists()
print("child terminated")
''')
    assert result == "exit_code=0\nchild terminated", result


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink"])
def test_summary_collector_rejects_unsafe_artifacts(tmp_path, kind):
    sandbox = PythonSandbox(tmp_path)
    target = tmp_path / "private.txt"
    target.write_text("never-copy-this")
    summary = sandbox.analysis / "analysis_summary.txt"
    if kind == "symlink":
        summary.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(summary)
    else:
        os.link(target, summary)
    with pytest.raises((OSError, ValueError)):
        sandbox.collect_summary()
    assert not (tmp_path / "analysis_summary.txt").exists()


def test_summary_destination_cannot_overwrite_an_existing_file(tmp_path):
    sandbox = PythonSandbox(tmp_path)
    (sandbox.analysis / "analysis_summary.txt").write_text("new")
    target = tmp_path / "private.txt"
    target.write_text("protected")
    (tmp_path / "analysis_summary.txt").symlink_to(target)
    with pytest.raises(FileExistsError):
        sandbox.collect_summary()
    assert target.read_text() == "protected"


@pytest.mark.parametrize("failure", ["missing", "namespace", "timeout"])
def test_isolation_failure_prevents_model_requests(tmp_path, monkeypatch, failure):
    import onc_co_scientist.harness.python_sandbox as module

    (tmp_path / "metadata.json").write_text(json.dumps(dict(dataset_id="d", max_iterations=1)))
    agent = runner(tmp_path)
    requests = []
    monkeypatch.setattr(agent, "_request", lambda *_: requests.append(True))
    if failure == "missing":
        monkeypatch.setattr(module.shutil, "which", lambda _: None)
    elif failure == "namespace":
        monkeypatch.setattr(
            module,
            "run_subprocess_in_group",
            lambda *a, **kw: subprocess.CompletedProcess([], 1, "", "namespace denied"),
        )
    else:

        def timed_out(*a, **kw):
            raise subprocess.TimeoutExpired("bwrap", 15)

        monkeypatch.setattr(module, "run_subprocess_in_group", timed_out)
    with pytest.raises(SandboxUnavailable):
        agent.run()
    assert requests == []
    assert not (tmp_path / "filesystem_isolation.json").exists()


def test_symlinked_inputs_and_analysis_directory_are_rejected(tmp_path):
    ws = tmp_path / "assigned"
    ws.mkdir()
    (ws / "analysis").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SandboxUnavailable, match="Analysis directory"):
        PythonSandbox(ws)
    (ws / "analysis").unlink()
    (ws / "metadata.json").symlink_to(tmp_path / "private.json")
    with pytest.raises(SandboxUnavailable, match="Public input"):
        PythonSandbox(ws).verify()
