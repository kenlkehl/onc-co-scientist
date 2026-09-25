#!/usr/bin/env bash
# Dedicated GCP G4 host: eight independent, loopback-only Gemma replicas.
set -euo pipefail
exec > >(tee -a /var/log/gemma-federation-startup.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
install -d /opt/gemma-federation /var/cache/gemma-hf
cd /opt/gemma-federation
apt-get update
apt-get install -y docker.io curl ca-certificates gnupg
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee gpu-inventory.csv
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8

IMAGE='vllm/vllm-openai@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1'
MODEL='nvidia/Gemma-4-31B-IT-NVFP4'
REVISION='4135a98a9b728a548947683219633b25682223ac'
docker pull "$IMAGE"
docker image inspect "$IMAGE" > image-manifest.json
curl -fSL https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/examples/tool_chat_template_gemma4.jinja -o tool_chat_template_gemma4.jinja
sha256sum tool_chat_template_gemma4.jinja > template.sha256
# Download once before loading eight replicas. No Hugging Face token is needed.
docker run --rm --entrypoint python3 -v /var/cache/gemma-hf:/root/.cache/huggingface "$IMAGE" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', revision='$REVISION')"
for gpu in $(seq 0 7); do
  port=$((8000 + gpu))
  if docker container inspect "gemma-federation-$gpu" >/dev/null 2>&1; then
    docker start "gemma-federation-$gpu"
    continue
  fi
  docker run -d --name "gemma-federation-$gpu" --restart=on-failure:3 \
    --gpus "device=$gpu" --ipc=host --network=host \
    --log-opt max-size=50m --log-opt max-file=3 \
    -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=8 \
    -v /var/cache/gemma-hf:/root/.cache/huggingface \
    -v /opt/gemma-federation:/deployment:ro \
    "$IMAGE" "$MODEL" --revision "$REVISION" --served-model-name gemma4-31b gemma4-31b-nvfp4 \
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 \
    --seed "$((20260921 + gpu))" \
    --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching --async-scheduling \
    --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 \
    --chat-template /deployment/tool_chat_template_gemma4.jinja \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --generation-config vllm \
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64}' \
    --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
done
date -Is > containers-started.txt
