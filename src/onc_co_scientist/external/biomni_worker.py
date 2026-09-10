"""Agent-side entrypoint. Mount this file alone, never the evaluator package.

All generated code and checkpoint deserialization execute inside the same OS
sandbox. Checkpoints are never deserialized by the trusted evaluator.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path

CONFIG = {}


class ClosedFileReference:
    """A closed handle stays closed on restore; never reopen an agent path.

    This preserves metadata and the unusable-resource behavior, not concrete
    IO class identity. Live handles remain unsupported and fail checkpointing.
    """

    closed = True

    def __init__(self, metadata):
        self.__dict__.update(metadata)

    def close(self):
        pass

    def _closed(self, *args, **kwargs):
        raise ValueError("I/O operation on closed file.")

    read = readline = readlines = write = writelines = seek = tell = flush = fileno = _closed
    readable = writable = seekable = isatty = truncate = __enter__ = __iter__ = __next__ = _closed

    def __exit__(self, *args):
        return None


def dump_namespace(namespace):
    import cloudpickle

    class NamespacePickler(cloudpickle.CloudPickler):
        def reducer_override(self, obj):
            if isinstance(obj, io.IOBase):
                if not obj.closed:
                    raise TypeError("Live file handles cannot be safely checkpointed")
                metadata = {}
                for name in ("name", "mode", "encoding", "errors"):
                    with suppress(AttributeError, ValueError):
                        metadata[name] = getattr(obj, name)
                return ClosedFileReference, (metadata,)
            return super().reducer_override(obj)

    buffer = io.BytesIO()
    NamespacePickler(buffer).dump(namespace)
    return buffer.getvalue()


def rpc(route, payload):
    request = urllib.request.Request(
        CONFIG["broker"] + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + CONFIG["secret"]},
    )
    with urllib.request.urlopen(request, timeout=CONFIG["request_timeout"]) as response:
        return json.load(response)


def benchmark_exchange(request: dict) -> dict:
    """Exchange with the benchmark. Supply request_id, round, action, payload.

    Actions: state, register (proposals), analyze (run_analyses: H references),
    assess (assessments), validate (validate: H reference), prepare_close ({}),
    close (optional assessments and narrative). Responses contain the public
    ledger, current round and due assessments. Print responses to inspect them.
    Use a unique request_id per action; retry identical requests with the same ID.
    """
    try:
        result = rpc("/exchange", request)
        # Biomni truncates observations at 10K characters. Keep full public
        # receipts on disk, with due obligations visible in the short response.
        if len(json.dumps(result)) > 7000:
            key = hashlib.sha256(str(request["request_id"]).encode()).hexdigest()
            path = Path("/work/benchmark_responses") / f"{key}.json"
            path.parent.mkdir(exist_ok=True)
            save(path, result)
            return {
                **{k: v for k, v in result.items() if k not in {"event", "claims"}},
                "claim_count": len(result.get("claims", [])),
                "full_public_response_file": str(path),
                "instruction": "Read this JSON file for the complete claim ledger and results.",
            }
        return result
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=str))
    temporary.replace(path)


def install_llm():
    import biomni.llm
    from biomni.config import default_config
    from langchain_openai import ChatOpenAI

    default_config.llm = CONFIG["model"]
    default_config.source = "Custom"
    default_config.base_url = CONFIG["broker"] + "/v1"
    default_config.api_key = CONFIG["secret"]
    default_config.temperature = CONFIG["temperature"]
    default_config.path = "/assets"
    default_config.timeout_seconds = CONFIG["tool_timeout"]

    def local_llm(model=None, temperature=None, stop_sequences=None, **kwargs):
        # Explicitly override even helper-specific default model/provider choices.
        return ChatOpenAI(
            model=CONFIG["model"],
            base_url=default_config.base_url,
            api_key=CONFIG["secret"],
            temperature=CONFIG["temperature"] if temperature is None else temperature,
            max_tokens=CONFIG["max_tokens"],
            timeout=CONFIG["request_timeout"],
            max_retries=0,
            stop=stop_sequences,
            reasoning_effort="xhigh",
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}
            },
        )

    original = biomni.llm.get_llm
    biomni.llm.get_llm = local_llm
    for name, module in list(sys.modules.items()):
        if name.startswith("biomni.") and getattr(module, "get_llm", None) is original:
            module.get_llm = local_llm


def main():
    global CONFIG
    CONFIG = json.loads(Path(sys.argv[1]).read_text())
    os.chdir("/work")

    def network_audit(event, args):
        if event == "socket.connect":
            with Path("/work/network_events.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"event": event, "destination": str(args[1]), "time": time.time()})
                    + "\n"
                )

    sys.addaudithook(network_audit)
    install_llm()
    import cloudpickle
    import numpy as np
    from biomni.agent import A1

    # Avoid displaying the broker credential in Biomni's configuration banner.
    from biomni.config import default_config
    from biomni.tool import support_tools

    # A1 instructs agents to inspect tools by their importable dotted name.
    # Injection alone is insufficient for that native discovery mechanism.
    support_tools.benchmark_exchange = benchmark_exchange
    benchmark_exchange.__module__ = "biomni.tool.support_tools"
    from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict

    original_to_dict = default_config.to_dict
    default_config.to_dict = lambda: {**original_to_dict(), "api_key": "[redacted]"}
    agent = A1(expected_data_lake_files=[])
    agent._custom_functions = {"benchmark_exchange": benchmark_exchange}
    agent._custom_tools = {
        "benchmark_exchange": {
            "name": "benchmark_exchange",
            "description": benchmark_exchange.__doc__,
            "module": "biomni.tool.support_tools",
        }
    }
    agent.configure()
    agent._inject_custom_functions_to_repl()
    assert support_tools._persistent_namespace["benchmark_exchange"] is benchmark_exchange
    source = support_tools.read_function_source_code("biomni.tool.support_tools.benchmark_exchange")
    if "def benchmark_exchange" not in source:
        raise RuntimeError("Benchmark tool is not inspectable by native Biomni tools")
    agent.critic_count = 0
    agent.user_task = CONFIG["prompt"]
    checkpoint = Path("/work/native_checkpoint.json")
    namespace_path = Path("/work/native_namespace.pkl")
    if CONFIG.get("resume"):
        saved = json.loads(checkpoint.read_text())
        if not saved["resumable"]:
            raise RuntimeError("Unsafe native checkpoint; code will not be replayed")
        restored = cloudpickle.loads(namespace_path.read_bytes())
        support_tools._persistent_namespace = restored["namespace"]
        random.setstate(restored["random_state"])
        np.random.set_state(restored["numpy_random_state"])
        os.chdir(saved["working_directory"])
        agent.system_prompt = saved["system_prompt"]
        inputs = {"messages": messages_from_dict(saved["messages"]), "next_step": "generate"}
    else:
        if agent.use_tool_retriever:
            selected = agent._prepare_resources_for_retrieval(CONFIG["prompt"])
            agent.update_system_prompt_with_selected_resources(selected)
        inputs = {"messages": [HumanMessage(content=CONFIG["prompt"])], "next_step": None}
    # Always visible even if the native resource retriever omits benchmark tooling.
    agent.system_prompt += (
        "\nThe Python function benchmark_exchange(request) is available. "
        + benchmark_exchange.__doc__
    )
    started = time.time()
    nodes = 0
    final_state = inputs
    for update in agent.app.stream(
        inputs,
        stream_mode="updates",
        config={"recursion_limit": 10000, "configurable": {"thread_id": CONFIG["run_id"]}},
    ):
        for node, state in update.items():
            nodes += 1
            final_state = state
            audit = rpc("/checkpoint", {})
            stable = node == "execute" or state.get("next_step") != "execute"
            saved = {
                "messages": messages_to_dict(state["messages"]),
                "system_prompt": agent.system_prompt,
                "node": node,
                "nodes": nodes,
                "resumable": False,
                "working_directory": os.getcwd(),
                **audit,
            }
            # Publish unsafe first: a crash between state/namespace publication
            # must never resume with inconsistent Python and conversation state.
            save(checkpoint, saved)
            if stable:
                try:
                    namespace = {
                        k: v
                        for k, v in support_tools._persistent_namespace.items()
                        if k not in {"benchmark_exchange", "__builtins__"}
                    }
                    temporary = namespace_path.with_suffix(".tmp")
                    temporary.write_bytes(
                        dump_namespace(
                            {
                                "namespace": namespace,
                                "random_state": random.getstate(),
                                "numpy_random_state": np.random.get_state(),
                            }
                        )
                    )
                    temporary.replace(namespace_path)
                    saved["resumable"] = True
                except Exception as exc:
                    saved["checkpoint_error"] = str(exc)
                save(checkpoint, saved)
            with Path("/work/native_events.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "node": node,
                            "messages": messages_to_dict(state["messages"]),
                            "time": time.time(),
                            **audit,
                        },
                        default=str,
                    )
                    + "\n"
                )
    agent._conversation_state = final_state
    save(
        Path("/work/native_result.json"),
        {
            "duration_seconds": time.time() - started,
            "nodes": nodes,
            "final_text": final_state["messages"][-1].content,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if Path("/work").exists():
            save(Path("/work/native_failure.json"), {"type": type(exc).__name__, "error": str(exc)})
        raise
