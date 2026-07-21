#!/usr/bin/env python3
"""End-to-end reproduction of the vLLM custom-all-reduce ROCm crash

    Failed: Cuda error .../csrc/custom_all_reduce_hip.cuh:457 'invalid argument'

through vLLM's *actual* code path, i.e. the same one hit by

    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

Minimal trigger conditions (all required):
  * ROCm, >= 2 fully-connected (XGMI) GPUs   -> vLLM enables custom all-reduce
  * tensor_parallel_size >= 2                -> custom all-reduce actually runs
  * CUDA graph capture enabled (enforce_eager=False, the default)
       -> vLLM records graph buffers and calls hipIpcGetMemHandle() on them
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
       -> those buffers are VMM-backed and NOT legacy-IPC-capable -> crash

The crash happens during engine init / CUDA-graph capture (weight load already
done), inside CustomAllreduce.register_graph_buffers ->
ops.get_graph_buffer_ipc_meta -> hipIpcGetMemHandle (custom_all_reduce_hip.cuh:457).

Run it:
    python repro_vllm_tp2_expandable_segments.py                 # reproduces (crashes)
    NO_EXPANDABLE=1 python repro_vllm_tp2_expandable_segments.py # control: should pass
    MODEL=/path/to/model TP=2 python repro_vllm_tp2_expandable_segments.py

Compared to repro_custom_ar_expandable_segments.py (single-GPU unit test that
isolates the failing HIP call), this script exercises the full vLLM stack, so
use it to confirm a fix end-to-end.
"""

import os
import sys

# MUST be set before torch/vllm initialize the CUDA/HIP caching allocator.
if os.environ.get("NO_EXPANDABLE") == "1":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Keep it offline/fast if a local HF cache exists; harmless otherwise.
os.environ.setdefault("VLLM_USE_V1", "1")

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TP = int(os.environ.get("TP", "2"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.3"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))


def main():
    print(f"[repro] MODEL={MODEL}")
    print(f"[repro] tensor_parallel_size={TP}")
    print(f"[repro] PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print("[repro] enforce_eager=False (CUDA graph capture ON)")
    print("[repro] disable_custom_all_reduce=False (custom all-reduce ON)")

    from vllm import LLM, SamplingParams

    # enforce_eager=False keeps CUDA-graph capture on (needed to hit the bug).
    # disable_custom_all_reduce=False keeps vLLM's custom all-reduce on.
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        enforce_eager=False,
        disable_custom_all_reduce=False,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_LEN,
        trust_remote_code=True,
    )

    out = llm.generate(["Hello, world!"], SamplingParams(max_tokens=8))
    print("[repro] generate OK (did NOT reproduce the crash):")
    print("       ", out[0].outputs[0].text.replace("\n", " ")[:80])
    print("[repro] -> custom all-reduce graph registration succeeded on this build/config")


if __name__ == "__main__":
    import torch

    n = torch.cuda.device_count()
    if n < TP:
        print(f"[skip] need >= {TP} GPUs, found {n}")
        sys.exit(2)
    main()
