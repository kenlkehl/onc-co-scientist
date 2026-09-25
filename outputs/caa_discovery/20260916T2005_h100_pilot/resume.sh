#!/usr/bin/env bash
set -euo pipefail
exec /data1/ken/envs/gptoss3/bin/python \
  /data1/ken/onc-co-scientist/outputs/caa_discovery/20260916T2005_h100_pilot/runtime/source/scripts/expected_surprising/run_local_caa_pilot.py \
  --resume /data1/ken/onc-co-scientist/outputs/caa_discovery/20260916T2005_h100_pilot/runtime "$@"
