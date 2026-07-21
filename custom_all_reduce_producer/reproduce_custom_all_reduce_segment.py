#!/usr/bin/env python3
"""Minimal unit reproduction of the vLLM custom-all-reduce crash on ROCm:

    Failed: Cuda error .../csrc/custom_all_reduce_hip.cuh:457 'invalid argument'

WHAT TRIGGERS IT
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" bash \
        examples/grpo_trainer/run_qwen3_8b_fsdp.sh

ROOT CAUSE (verified by this script)
    `expandable_segments:True` makes the PyTorch caching allocator back device
    memory with the HIP virtual-memory API (hipMemCreate / hipMemMap /
    hipMemAddressReserve). That memory is NOT "legacy HIP IPC capable"
    (HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE == 0).

    During CUDA-graph capture, vLLM's custom all-reduce records the activation
    tensors it reduces (graph_unreg_buffers_) and, at the end of capture, tries
    to export an IPC handle for each of them:

        csrc/custom_all_reduce_hip.cuh  (get_graph_buffer_ipc_meta)
          453:  hipPointerGetAttribute(&base_ptr, RANGE_START_ADDR, ptr)
          456:  hipIpcGetMemHandle((hipIpcMemHandle_t*)&handles[...], base_ptr)  <-- line 457

    hipIpcGetMemHandle() only works on plain hipMalloc memory, so on a
    VMM-backed (expandable-segments) pointer it returns hipErrorInvalidValue,
    whose string is exactly "invalid argument".

WHAT THIS SCRIPT DOES
    Reproduces ONLY that failing HIP call, on a SINGLE GPU, with no vLLM, no
    torch.distributed and no model download. It allocates a tensor through the
    PyTorch caching allocator (so it honors expandable_segments) and then makes
    the exact same two HIP driver calls vLLM makes at lines 453/456-457.

USAGE
    # run both allocator configs and show the contrast (default):
    python repro_custom_ar_expandable_segments.py

    # run a single config (what the harness spawns internally):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python repro_custom_ar_expandable_segments.py --probe
"""

import ctypes
import os
import re
import subprocess
import sys

# hipPointer_attribute values (rocm .../include/hip/driver_types.h)
HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10
HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11

EXIT_OK = 0        # hipIpcGetMemHandle succeeded
EXIT_NA = 2        # not applicable (no GPU / not a ROCm build)
EXIT_REPRO = 3     # hipIpcGetMemHandle failed -> crash reproduced


def _load_already_loaded_hip():
    """Attach to the libamdhip64 that torch already dlopened.

    Loading a *second* copy of libamdhip64 (e.g. ctypes.CDLL("libamdhip64.so"))
    re-runs ROCm/LLVM static initializers and aborts with
    "Option 'spirv-expand-step' registered more than once", so we reuse the
    exact mapped path from /proc/self/maps instead.
    """
    libpath = None
    with open("/proc/self/maps") as fh:
        for line in fh:
            m = re.search(r"(\S*libamdhip64\.so\S*)", line)
            if m:
                libpath = m.group(1)
                break
    if libpath is None:
        return None, None
    hip = ctypes.CDLL(libpath)
    hip.hipGetErrorString.restype = ctypes.c_char_p
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipPointerGetAttribute.restype = ctypes.c_int
    hip.hipPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    hip.hipIpcGetMemHandle.restype = ctypes.c_int
    hip.hipIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return hip, libpath


def probe():
    import torch

    if not torch.cuda.is_available():
        print("[skip] no GPU visible to torch")
        return EXIT_NA
    if getattr(torch.version, "hip", None) is None:
        print("[skip] this torch is not a ROCm/HIP build; repro targets ROCm")
        return EXIT_NA

    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")

    # Allocate through the caching allocator so expandable_segments applies.
    # 8 MiB matches vLLM's custom-AR buffer size and is large enough to land in
    # an expandable segment.
    t = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    ptr = t.data_ptr()

    hip, libpath = _load_already_loaded_hip()
    if hip is None:
        print("[skip] libamdhip64.so not found in process maps")
        return EXIT_NA

    def es(err):
        return hip.hipGetErrorString(err).decode()

    print(f"[hiplib] {libpath}")
    print(f"[config] PYTORCH_CUDA_ALLOC_CONF={conf}")
    print(f"[alloc ] torch tensor data_ptr=0x{ptr:x}")

    # vLLM line 453: get the allocation's base address.
    base = ctypes.c_void_p()
    e = hip.hipPointerGetAttribute(
        ctypes.byref(base), HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR, ctypes.c_void_p(ptr)
    )
    print(f"[attr  ] RANGE_START_ADDR           -> {e} ({es(e)}) base=0x{base.value or 0:x}")

    cap = ctypes.c_int(-1)
    hip.hipPointerGetAttribute(
        ctypes.byref(cap), HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE, ctypes.c_void_p(ptr)
    )
    print(f"[attr  ] IS_LEGACY_HIP_IPC_CAPABLE  -> {cap.value}   (0 = cannot use hipIpcGetMemHandle)")

    # vLLM lines 456-457: export an IPC handle for that base address.
    handle = (ctypes.c_char * 64)()
    base_for_ipc = base.value if base.value else ptr
    e3 = hip.hipIpcGetMemHandle(handle, ctypes.c_void_p(base_for_ipc))
    if e3 != 0:
        # Reproduce vLLM's CUDACHECK message format verbatim.
        print(f"Failed: Cuda error csrc/custom_all_reduce_hip.cuh:457 '{es(e3)}'")
        print("[VERDICT] REPRODUCED: hipIpcGetMemHandle failed on expandable-segments memory")
        return EXIT_REPRO
    print("[VERDICT] hipIpcGetMemHandle succeeded (IPC handle exported)")
    return EXIT_OK


def run_both():
    here = os.path.abspath(__file__)
    dev = os.environ.get("HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    dev = dev.split(",")[0] if dev else "0"
    results = {}
    for val in ("expandable_segments:True", "expandable_segments:False"):
        print("=" * 76)
        print(f"###  PYTORCH_CUDA_ALLOC_CONF={val}   (HIP_VISIBLE_DEVICES={dev})")
        print("=" * 76)
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = val
        env["HIP_VISIBLE_DEVICES"] = dev
        results[val] = subprocess.call([sys.executable, here, "--probe"], env=env)
        print()

    on = results.get("expandable_segments:True")
    off = results.get("expandable_segments:False")
    print("=" * 76)
    print(f"summary: expandable_segments:True -> exit {on}, "
          f"expandable_segments:False -> exit {off}")
    if on == EXIT_REPRO and off == EXIT_OK:
        print("RESULT: crash REPRODUCED (fails only with expandable_segments:True)")
        return 0
    if on == EXIT_NA or off == EXIT_NA:
        print("RESULT: inconclusive (see [skip] messages above)")
        return 2
    print("RESULT: NOT reproduced as expected on this platform")
    return 1


if __name__ == "__main__":
    if "--probe" in sys.argv:
        sys.exit(probe())
    sys.exit(run_both())
