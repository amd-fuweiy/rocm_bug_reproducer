#!/usr/bin/env bash
# Usage: run.sh <image> [extra docker -e args...] -- [repro.py args...]
# Example: run.sh rocm/primus:v26.7
#          run.sh rocm/primus:v26.7 -e NCCL_P2P_DISABLE=1
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMG="$1"; shift
DOCKER_ARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do DOCKER_ARGS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
docker run --rm --network host --ipc host --shm-size 8g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  "${DOCKER_ARGS[@]}" -v "$HERE:/repro:ro" -w /repro --entrypoint bash "$IMG" \
  -c 'unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; python3 repro.py "$@"' _ "$@"

