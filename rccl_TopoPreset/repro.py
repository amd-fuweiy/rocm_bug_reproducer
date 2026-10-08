#!/usr/bin/env python3
"""RCCL TopoPreset mini reproducer (F011).

N processes on one node, process i sees only GPU i (HIP_VISIBLE_DEVICES=i), so every process uses
local ordinal 0. Each runs init_process_group("nccl") and one all_reduce. Peer-to-peer is left on.

  RCCL 2.30.4: all ranks print all_reduce=N.
  RCCL 2.30.7: ranks 1..N-1 fail with
      ncclInternalError ... TopoPreset: Local rank <r> not found in intra-node graph.

Usage: python3 repro.py [--nproc 8] [--timeout 60]
Exit code: 0 = all ranks ok, 1 = a rank failed, 124 = timed out.
"""
import argparse
import os
import subprocess
import sys

WORKER = r'''
import os, torch, torch.distributed as dist
r, n = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
try:
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=r, world_size=n)
    t = torch.ones(1, device="cuda:0")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    ok = t.item() == n
    print(f"rank {r} pci_bus={torch.cuda.get_device_properties(0).pci_bus_id} all_reduce={t.item():.0f} {'OK' if ok else 'WRONG'}", flush=True)
    os._exit(0 if ok else 1)
except Exception as e:
    last = [l for l in str(e).splitlines() if l.strip()][-1]
    print(f"rank {r} FAIL {type(e).__name__}: {last}", flush=True)
    os._exit(1)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args()

    import ctypes
    import torch
    # torch.cuda.nccl.version() is the header version torch was compiled against, not the library
    # that is loaded, so ask the loaded librccl itself.
    libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "librccl.so" in l})
    runtime = []
    for path in libs:
        n = ctypes.c_int()
        ctypes.CDLL(path).ncclGetVersion(ctypes.byref(n))
        runtime.append(f"{path} ({os.path.getsize(path)} B) -> {n.value}")
    print(f"torch {torch.__version__}  hip {torch.version.hip}  "
          f"rccl(torch header) {'.'.join(map(str, torch.cuda.nccl.version()))}  "
          f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '<unset>')}", flush=True)
    for line in runtime or ["<librccl not mapped>"]:
        print(f"rccl(loaded) {line}", flush=True)

    procs = []
    for r in range(a.nproc):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        env.update(HIP_VISIBLE_DEVICES=str(r), RANK=str(r), WORLD_SIZE=str(a.nproc),
                   LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29931")
        procs.append(subprocess.Popen([sys.executable, "-c", WORKER], env=env,
                                      stderr=subprocess.DEVNULL))
    rc = 0
    for p in procs:
        try:
            rc |= p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            rc = 124
    for p in procs:
        if p.poll() is None:
            p.kill()
    print("RESULT:", {0: "PASS", 124: "HANG"}.get(rc, "FAIL"), flush=True)
    return 124 if rc == 124 else (0 if rc == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())

