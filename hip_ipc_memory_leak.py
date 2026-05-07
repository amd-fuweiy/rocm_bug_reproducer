import ctypes
import sys

import torch


class HipIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def load_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name), name
        except OSError:
            continue
    raise RuntimeError("Cannot load HIP runtime library")


def run_hip_ipc_cycle(iterations=10, alloc_mb=500):
    hip, lib_name = load_hip()

    hip_ptr_t = ctypes.c_void_p
    hip.hipMalloc.argtypes = [ctypes.POINTER(hip_ptr_t), ctypes.c_size_t]
    hip.hipMalloc.restype = ctypes.c_int
    hip.hipFree.argtypes = [hip_ptr_t]
    hip.hipFree.restype = ctypes.c_int
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(HipIpcMemHandle), hip_ptr_t]
    hip.hipIpcGetMemHandle.restype = ctypes.c_int
    hip.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    hip.hipMemGetInfo.restype = ctypes.c_int

    def mem_free_mb():
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        err = hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if err != 0:
            raise RuntimeError(f"hipMemGetInfo failed: err={err}")
        return free.value / (1024**2)

    base = mem_free_mb()
    per_step = []
    alloc_bytes = alloc_mb * 1024 * 1024

    for i in range(iterations):
        ptr = hip_ptr_t()
        err = hip.hipMalloc(ctypes.byref(ptr), alloc_bytes)
        if err != 0:
            raise RuntimeError(f"hipMalloc failed at iter {i}: err={err}")
        handle = HipIpcMemHandle()
        err = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
        if err != 0:
            raise RuntimeError(f"hipIpcGetMemHandle failed at iter {i}: err={err}")
        err = hip.hipFree(ptr)
        if err != 0:
            raise RuntimeError(f"hipFree failed at iter {i}: err={err}")
        now = mem_free_mb()
        per_step.append(now)
        print(f"[HIP IPC] step {i:02d}: free={now:.2f} MB")

    delta = base - per_step[-1]
    return lib_name, base, per_step[-1], delta


def run_torch_alloc_control(iterations=20, numel=50_000_000):
    # Control group: no IPC API involved.
    dev = torch.device("cuda:0")
    torch.cuda.empty_cache()
    f0, t0 = torch.cuda.mem_get_info(dev)
    used0 = (t0 - f0) / (1024**3)

    used = used0
    for _ in range(iterations):
        x = torch.empty(numel, dtype=torch.float32, device=dev)
        del x
        torch.cuda.empty_cache()
        f, t = torch.cuda.mem_get_info(dev)
        used = (t - f) / (1024**3)

    return used - used0


def main():
    print("=== Environment ===")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch.version.hip={torch.version.hip}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")

    print("\n=== Control: torch alloc/free ===")
    control_growth_gb = run_torch_alloc_control()
    print(f"control_growth={control_growth_gb:.2f} GB")

    print("\n=== Reproducer: HIP IPC handle cycle ===")
    lib_name, free_start, free_end, delta_mb = run_hip_ipc_cycle()
    print(f"hip_lib={lib_name}")
    print(f"free_start={free_start:.2f} MB, free_end={free_end:.2f} MB")
    print(f"free_delta={delta_mb:.2f} MB")

    # Threshold: > 2GB drop after 10*500MB cycles indicates severe leak.
    leaked = delta_mb > 2048
    print("\n=== Classification ===")
    if leaked and abs(control_growth_gb) < 0.8:
        print("HIP/runtime-level IPC memory release bug reproduced.")
    elif leaked:
        print("Leak reproduced, but control is noisy; likely HIP/runtime-level issue.")
    else:
        print("No obvious HIP IPC leak reproduced.")


if __name__ == "__main__":
    main()
