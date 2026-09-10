#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:/home/kenneth_kehl/vllm/bin:$PATH"
exec /home/kenneth_kehl/vllm/bin/vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port 8000 --max-model-len 262144 \
  --gpu-memory-utilization 0.8 --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}' "$@"
