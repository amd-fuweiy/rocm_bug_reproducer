import argparse
import ctypes
import gc
import io
import multiprocessing as mp
import pickle
import sys
import traceback

import torch
from multiprocessing.reduction import ForkingPickler


def load_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name), name
        except OSError:
            continue
    raise RuntimeError("Cannot load HIP runtime library")


def hip_used_gb(hip):
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    hip.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    hip.hipMemGetInfo.restype = ctypes.c_int
    err = hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if err != 0:
        raise RuntimeError(f"hipMemGetInfo failed: err={err}")
    return (total.value - free.value) / (1024**3)


def run_hip_baseline(hip, iterations=8, alloc_mb=500):
    class HipIpcMemHandle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_char * 64)]

    hip_ptr_t = ctypes.c_void_p
    hip.hipMalloc.argtypes = [ctypes.POINTER(hip_ptr_t), ctypes.c_size_t]
    hip.hipMalloc.restype = ctypes.c_int
    hip.hipFree.argtypes = [hip_ptr_t]
    hip.hipFree.restype = ctypes.c_int
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(HipIpcMemHandle), hip_ptr_t]
    hip.hipIpcGetMemHandle.restype = ctypes.c_int

    start = hip_used_gb(hip)
    for i in range(iterations):
        ptr = hip_ptr_t()
        err = hip.hipMalloc(ctypes.byref(ptr), alloc_mb * 1024 * 1024)
        if err != 0:
            raise RuntimeError(f"hipMalloc failed at iter {i}: err={err}")
        handle = HipIpcMemHandle()
        err = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
        if err != 0:
            raise RuntimeError(f"hipIpcGetMemHandle failed at iter {i}: err={err}")
        err = hip.hipFree(ptr)
        if err != 0:
            raise RuntimeError(f"hipFree failed at iter {i}: err={err}")
    end = hip_used_gb(hip)
    growth = end - start
    print(f"[HIP baseline] used: {start:.2f} -> {end:.2f} GB (growth={growth:.2f} GB)")
    return growth


def _worker_loop(recv_conn, ack_conn):
    try:
        while True:
            payload = recv_conn.recv_bytes()
            if payload == b"__STOP__":
                ack_conn.send(("ok", None))
                break
            obj = pickle.loads(payload)
            _ = obj[0].item()
            del obj
            gc.collect()
            torch.cuda.empty_cache()
            ack_conn.send(("ok", None))
    except Exception:
        ack_conn.send(("err", traceback.format_exc()))


def run_no_ray_trigger(hip, iterations=10, numel=125_000_000):
    mp.set_start_method("spawn", force=True)
    child_recv, parent_send = mp.Pipe(duplex=False)
    parent_recv_ack, child_send_ack = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker_loop, args=(child_recv, child_send_ack))
    p.start()
    child_recv.close()
    child_send_ack.close()

    start = hip_used_gb(hip)
    print(f"[No-Ray trigger] hip used start: {start:.2f} GB")
    for i in range(iterations):
        t = torch.empty(numel, dtype=torch.float32, device="cuda:0")
        buf = io.BytesIO()
        ForkingPickler(buf, -1).dump(t)
        payload = buf.getvalue()

        parent_send.send_bytes(payload)
        status, info = parent_recv_ack.recv()
        if status != "ok":
            raise RuntimeError(f"worker failed at step {i}:\n{info}")

        del t
        del payload
        del buf
        gc.collect()
        torch.cuda.empty_cache()

        now = hip_used_gb(hip)
        print(f"[No-Ray trigger] step {i:02d}: hip used={now:.2f} GB")

    parent_send.send_bytes(b"__STOP__")
    status, info = parent_recv_ack.recv()
    if status != "ok":
        raise RuntimeError(f"worker stop failed:\n{info}")
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        p.join()
        raise RuntimeError("worker did not exit cleanly")

    end = hip_used_gb(hip)
    growth = end - start
    return growth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--numel", type=int, default=125_000_000)
    parser.add_argument("--hip-iters", type=int, default=8)
    parser.add_argument("--hip-alloc-mb", type=int, default=500)
    args = parser.parse_args()

    hip, hip_lib = load_hip()
    print("=== Environment ===")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}, torch.version.hip={torch.version.hip}")
    print(f"hip_lib={hip_lib}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}, torch.cuda.device_count={torch.cuda.device_count()}")

    print("\n=== Test A: pure HIP IPC baseline ===")
    hip_growth = run_hip_baseline(hip, iterations=args.hip_iters, alloc_mb=args.hip_alloc_mb)

    print("\n=== Test B: no-Ray memory_leak-equivalent trigger ===")
    trigger_growth = run_no_ray_trigger(hip, iterations=args.iters, numel=args.numel)
    print(f"[No-Ray trigger] growth={trigger_growth:.2f} GB")

    print("\n=== Classification ===")
    trigger_leak = trigger_growth > 1.0
    hip_leak = hip_growth > 1.0
    if trigger_leak and not hip_leak:
        print("No-Ray trigger leaks while pure HIP baseline is clean: likely PyTorch IPC ref/ACK path issue.")
    elif trigger_leak and hip_leak:
        print("No-Ray trigger leaks and HIP baseline leaks: HIP/runtime issue is likely involved.")
    elif (not trigger_leak) and (not hip_leak):
        print("No leak reproduced in either path.")
    else:
        print("Inconclusive mixed result.")


if __name__ == "__main__":
    main()
