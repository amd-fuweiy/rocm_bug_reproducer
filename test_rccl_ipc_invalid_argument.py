#!/usr/bin/env python3
"""
PyTorch + RCCL P2P bug on ROCm 7.13 / RCCL 2.28.3 — self-contained, no vLLM.

On ROCm 7.13 (RCCL version code 22803) the failure mode changed compared to the
old ROCm 7.0.2 stack:

  * Old stack: the trigger was the *init API* — legacy `ncclCommInitRank` failed
    with "invalid device ordinal" while `ncclCommInitRankConfig` worked around it.
  * New stack: the failure moved into the **P2P/IPC transport setup** and now
    happens for BOTH init APIs and even *without* PyTorch. `ncclCommInitRankConfig`
    no longer helps. The error is:
        [FATAL ERROR]: HIP failure: 'invalid argument'   (src/transport/p2p.cc)
    Root cause: under an ASYMMETRIC HIP_VISIBLE_DEVICES topology RCCL selects the
    P2P/IPC transport and then issues a HIP peer-access call referencing a peer
    GPU that is NOT visible inside the local process.

So the meaningful A/B/C comparison on ROCm 7.13 is no longer "init API"; it is
"visibility symmetry × P2P on/off":

  case A  asymmetric visibility + P2P enabled        -> EXPECT FAIL  (init, src/transport/p2p.cc 'invalid argument')
  case B  asymmetric visibility + NCCL_P2P_DISABLE=1 -> EXPECT FAIL  (init OK via SHM, but the
                                                                      collective then hits 'invalid device ordinal')
  case C  symmetric  visibility + P2P enabled        -> EXPECT PASS  (the real fix, keeps P2P/IPC)

  A vs B => same asymmetric topology, only P2P transport on/off. Disabling P2P merely
            MOVES the failure from comm-init to the collective; it is NOT a real fix.
  A/B vs C => only making the visibility symmetric makes the asymmetric topology work
            end-to-end. This is what verl / ROLL effectively do (consistent visibility).

Topologies (world=2, one node, all cases device_mgr=torch, init=ncclCommInitRank):
  asymmetric: rank0 HIP_VISIBLE_DEVICES=g0,g1 binds ordinal 1; rank1 =g2 binds ordinal 0
  symmetric : rank0 HIP_VISIBLE_DEVICES=g0    binds ordinal 0; rank1 =g1 binds ordinal 0

NOTE (ROCm 7.13 / torch 2.11): PyTorch must be imported BEFORE dlopen'ing
librccl.so.1, otherwise the two LLVM copies double-register the same cl::opt
('spirv-expand-step registered more than once') and the process aborts.

Run:
  python repro_torch_rccl.py --gpus 0,1,2

Env: ROCm 7.13 / RCCL 2.28.3 / PyTorch 2.11 rocm build. Requires 3 GPUs.
"""
import argparse, ctypes, multiprocessing, os, sys, time, traceback

NCCL_FLOAT32, NCCL_SUM, INT_MIN = 7, 0, -2147483648


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _load_rccl():
    r = ctypes.CDLL("librccl.so.1")
    r.ncclGetErrorString.restype = ctypes.c_char_p
    r.ncclGetErrorString.argtypes = [ctypes.c_int]
    r.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    r.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
    r.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ncclUniqueId, ctypes.c_int]
    r.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    r.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    return r


def _nck(r, ret, where):
    if ret != 0:
        raise RuntimeError(f"{where}: ncclResult={ret} ({r.ncclGetErrorString(ret).decode()})")


def _visibility(rank, gpus, sym):
    """Return (HIP_VISIBLE_DEVICES string, local ordinal to bind)."""
    if sym:
        return (f"{gpus[0]}" if rank == 0 else f"{gpus[1]}"), 0
    if rank == 0:
        return f"{gpus[0]},{gpus[1]}", 1
    return f"{gpus[2]}", 0


def _proc(rank, gpus, cfg, uid_bytes, qlog):
    sym = cfg["sym"]
    p2p_disable = cfg["p2p_disable"]
    cvd, loc = _visibility(rank, gpus, sym)
    os.environ["HIP_VISIBLE_DEVICES"] = cvd
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    if p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    else:
        os.environ.pop("NCCL_P2P_DISABLE", None)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    who = f"rank{rank}"

    try:
        # ROCm 7.13 / torch 2.11: import torch BEFORE dlopen'ing librccl, else the
        # two LLVM copies abort with 'spirv-expand-step registered more than once'.
        import torch
        torch.cuda.set_device(loc)
        visible = torch.cuda.device_count()
        r = _load_rccl()
        qlog.put(("log", f"[{who}] HIP_VISIBLE_DEVICES={cvd} set_device({loc}) "
                         f"visible={visible} P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '0')}"))

        uid = ncclUniqueId(); ctypes.memmove(ctypes.byref(uid), uid_bytes, 128)
        comm = ctypes.c_void_p()
        _nck(r, r.ncclCommInitRank(ctypes.byref(comm), 2, uid, rank), "ncclCommInitRank")
        qlog.put(("log", f"[{who}] ncclCommInitRank returned ncclSuccess"))

        buf = torch.full((1,), float(rank + 1), dtype=torch.float32, device=f"cuda:{loc}")
        stream = torch.cuda.current_stream().cuda_stream
        _nck(r, r.ncclAllReduce(ctypes.c_void_p(buf.data_ptr()), ctypes.c_void_p(buf.data_ptr()),
                                1, NCCL_FLOAT32, NCCL_SUM, comm, ctypes.c_void_p(stream)), "ncclAllReduce")
        torch.cuda.synchronize()
        val = buf.item()

        ok = abs(val - 3.0) < 1e-3
        qlog.put(("log", f"[{who}] allreduce={val:.1f} (expect 3.0) verify={'OK' if ok else 'MISMATCH'}"))
        qlog.put(("done" if ok else "err", None if ok else f"{who}: result mismatch {val}"))
        try:
            r.ncclCommDestroy(comm)
        except Exception:
            pass
    except Exception as e:
        qlog.put(("err", f"{who}: {type(e).__name__}: {e}\n{traceback.format_exc()}"))


def run_case(name, cfg, gpus, timeout):
    desc = (f"visibility={'symmetric' if cfg['sym'] else 'asymmetric'}, "
            f"P2P={'disabled' if cfg['p2p_disable'] else 'enabled'}")
    print(f"\n{'='*68}\n  CASE {name}: {desc}\n{'='*68}")
    r = _load_rccl()
    uid = ncclUniqueId()
    _nck(r, r.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
    uid_bytes = ctypes.string_at(ctypes.byref(uid), 128)
    ctx = multiprocessing.get_context("spawn")
    qlog = ctx.Queue()
    procs = [ctx.Process(target=_proc, args=(i, gpus, cfg, uid_bytes, qlog)) for i in range(2)]
    for p in procs: p.start()
    done, errs, t0 = 0, [], time.time()
    while done < 2 and time.time() - t0 < timeout:
        try:
            kind, data = qlog.get(timeout=1)
            if kind == "log": print("  " + data)
            elif kind == "done": done += 1
            elif kind == "err":
                if data: print(f"  X {data.splitlines()[0]}"); errs.append(data)
        except Exception:
            if all(not p.is_alive() for p in procs): break
    for p in procs:
        if p.is_alive(): p.terminate(); p.join(5)
        if p.is_alive(): p.kill()
    ok = (not errs) and done >= 2
    if errs:
        print("\n  --- first error ---\n  " + errs[0].splitlines()[0])
    print(f"  >>> CASE {name}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="0,1,2")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(gpus) < 3:
        print("need 3 GPUs, e.g. --gpus 0,1,2"); sys.exit(2)

    r = _load_rccl(); v = ctypes.c_int(); r.ncclGetVersion(ctypes.byref(v))
    print(f"RCCL version code = {v.value}")

    cases = [
        ("A", dict(sym=False, p2p_disable=False)),  # root cause: init fails in p2p.cc
        ("B", dict(sym=False, p2p_disable=True)),   # P2P off: init OK, collective fails -> insufficient
        ("C", dict(sym=True,  p2p_disable=False)),  # real fix: symmetric visibility
    ]
    res = {n: run_case(n, cfg, gpus, args.timeout) for n, cfg in cases}

    print(f"\n{'='*68}\n  SUMMARY\n{'='*68}")
    labels = {"A": "asymmetric + P2P enabled        ",
              "B": "asymmetric + NCCL_P2P_DISABLE=1 ",
              "C": "symmetric  + P2P enabled        "}
    for n in ("A", "B", "C"):
        print(f"  CASE {n}  {labels[n]} : {'PASS' if res[n] else 'FAIL'}")
    if (not res["A"]) and (not res["B"]) and res["C"]:
        print("\n  => Reproduced (ROCm 7.13). A and B fail, only C passes.")
        print("     A: asymmetric visibility breaks RCCL P2P transport setup (p2p.cc 'invalid argument').")
        print("     B: NCCL_P2P_DISABLE=1 only moves the failure to the collective ('invalid device")
        print("        ordinal') -> disabling P2P is NOT a real fix for an asymmetric topology.")
        print("     C: symmetric visibility (each rank binds ordinal 0) works end-to-end with P2P/IPC.")
        print("     Note: ncclCommInitRankConfig no longer helps on this stack (failure is transport-level).")
    sys.exit(0)


if __name__ == "__main__":
    main()
