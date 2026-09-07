#!/usr/bin/env python3
"""Gemini Vertex alternative using the shared sandboxed scientific agent controller."""
import sys

try:
    from scripts.vllm_cli_json_adapter import main
except ModuleNotFoundError:
    from vllm_cli_json_adapter import main

if __name__ == "__main__":
    raise SystemExit(main(["--provider", "gemini-vertex", *sys.argv[1:]]))
