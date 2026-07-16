#!/usr/bin/env python3
"""
Pure-HIP, dependency-free mini reproducer for a hang in the ROCm Virtual Memory Management
(VMM) driver API:

    hipMemSetAccess() on freshly hipMemCreate + hipMemMap'd memory  ->  HANGS FOREVER
    (observed on the ROCm 7.14.0 rc3 runtime; the call never returns).

This is the sibling defect to the hipIpcGetMemHandle(VMM) failure probed by
hip_vmm_ipc_probe.py. Every VMM setup step up to and including hipMemMap succeeds and returns
immediately; only the subsequent hipMemSetAccess never returns.

Like the sibling probe this uses ONLY ctypes + libamdhip64 (no torch, no vLLM, no ray, no
openrlhf), so it runs on ANY ROCm image and is ideal for cross-version regression testing.

The whole VMM sequence runs in a spawned child process guarded by a hard timeout. Each HIP
call is bracketed by a "before/after" progress marker sent to the parent, so if a call hangs
the parent can report exactly which one (expected: HANG(hipMemSetAccess)).

Exit code:
  0  -> hipMemSetAccess returned (no hang; may still be an error, printed as FAIL)
  2  -> BUG reproduced: hipMemSetAccess hung past the timeout
  3  -> VMM not supported / setup step failed / other (inconclusive)
  4  -> cannot load libamdhip64.so (inconclusive)
"""
import ctypes
import glob
import multiprocessing as mp
import os
import sys
import time

# How long (seconds) to wait for the VMM sequence before declaring a HANG.
TEST_TIMEOUT_S = 30

SIZE = 2 * 1024 * 1024  # 2 MiB

# ---- HIP enums / constants ----
hipMemAllocationTypePinned = 1
hipMemLocationTypeDevice = 1
hipMemAccessFlagsProtReadWrite = 3
hipMemAllocationGranularityRecommended = 1
hipMemHandleTypeNone = 0


def _candidate_lib_dirs():
    """Directories that may hold libamdhip64.so on wheel-based (TheRock) ROCm installs
    where the runtime is shipped inside python site-packages rather than on the loader path."""
    dirs = []
    for base in sys.path:
        if not base or not os.path.isdir(base):
            continue
        for pat in ("_rocm_sdk_devel/lib", "_rocm_sdk_core/lib", "torch/lib"):
            dirs.append(os.path.join(base, pat))
    return dirs


def load_hip():
    names = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5")
    # 1) plain names via the normal loader search path (respects LD_LIBRARY_PATH)
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    # 2) wheel-based ROCm: search site-packages for a concrete .so file
    for d in _candidate_lib_dirs():
        for name in names:
            for path in sorted(glob.glob(os.path.join(d, name + "*"))):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
    return None


class hipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class hipMemAllocationProp(ctypes.Structure):
    class _AllocFlags(ctypes.Structure):
        _fields_ = [
            ("compressionType", ctypes.c_ubyte),
            ("gpuDirectRDMACapable", ctypes.c_ubyte),
            ("usage", ctypes.c_ushort),
            ("reserved", ctypes.c_ubyte * 4),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", hipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class hipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", hipMemLocation), ("flags", ctypes.c_int)]


def _bind(HIP):
    for fn in ("hipMemGetAllocationGranularity", "hipMemCreate", "hipMemAddressReserve",
               "hipMemMap", "hipMemSetAccess", "hipMemUnmap", "hipMemAddressFree",
               "hipMemRelease", "hipRuntimeGetVersion", "hipDeviceGetName", "hipSetDevice",
               "hipGetErrorString"):
        if hasattr(HIP, fn):
            getattr(HIP, fn).restype = ctypes.c_int
    if hasattr(HIP, "hipGetErrorString"):
        HIP.hipGetErrorString.restype = ctypes.c_char_p


def _errstr(HIP, err):
    try:
        return HIP.hipGetErrorString(ctypes.c_int(err)).decode()
    except Exception:
        return "?"


def _mark(q, msg):
    """Send a timestamped progress marker to the parent so a hang can be pinpointed."""
    if q is not None:
        q.put(("step", f"{time.time():.3f} {msg}"))


def _vmm_setaccess_child(q):
    HIP = load_hip()
    if HIP is None:
        q.put(("result", "NO_HIP"))
        return
    if not hasattr(HIP, "hipMemCreate"):
        q.put(("result", "VMM_UNSUPPORTED(no hipMemCreate)"))
        return
    _bind(HIP)
    HIP.hipSetDevice(ctypes.c_int(0))

    prop = hipMemAllocationProp()
    prop.type = hipMemAllocationTypePinned
    prop.requestedHandleType = hipMemHandleTypeNone
    prop.location.type = hipMemLocationTypeDevice
    prop.location.id = 0

    granularity = ctypes.c_size_t(0)
    _mark(q, "-> hipMemGetAllocationGranularity")
    HIP.hipMemGetAllocationGranularity(
        ctypes.byref(granularity), ctypes.byref(prop),
        ctypes.c_int(hipMemAllocationGranularityRecommended),
    )
    gran = granularity.value or (2 * 1024 * 1024)
    size = ((SIZE + gran - 1) // gran) * gran
    _mark(q, f"<- granularity={gran} size={size}")

    gen_handle = ctypes.c_ulonglong(0)
    _mark(q, "-> hipMemCreate")
    e = HIP.hipMemCreate(ctypes.byref(gen_handle), ctypes.c_size_t(size), ctypes.byref(prop), ctypes.c_ulonglong(0))
    _mark(q, f"<- hipMemCreate ret={e}")
    if e != 0:
        q.put(("result", f"VMM_SETUP_FAIL(hipMemCreate={e}:{_errstr(HIP, e)})"))
        return

    dptr = ctypes.c_void_p()
    _mark(q, "-> hipMemAddressReserve")
    e = HIP.hipMemAddressReserve(ctypes.byref(dptr), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_ulonglong(0))
    _mark(q, f"<- hipMemAddressReserve ret={e} ptr={dptr.value}")
    if e != 0:
        HIP.hipMemRelease(gen_handle)
        q.put(("result", f"VMM_SETUP_FAIL(AddressReserve={e}:{_errstr(HIP, e)})"))
        return

    _mark(q, "-> hipMemMap")
    e = HIP.hipMemMap(dptr, ctypes.c_size_t(size), ctypes.c_size_t(0), gen_handle, ctypes.c_ulonglong(0))
    _mark(q, f"<- hipMemMap ret={e}")
    if e != 0:
        HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
        HIP.hipMemRelease(gen_handle)
        q.put(("result", f"VMM_SETUP_FAIL(hipMemMap={e}:{_errstr(HIP, e)})"))
        return

    desc = hipMemAccessDesc()
    desc.location.type = hipMemLocationTypeDevice
    desc.location.id = 0
    desc.flags = hipMemAccessFlagsProtReadWrite

    # ---- THE PROBE: this call is expected to hang forever on the affected runtime ----
    _mark(q, "-> hipMemSetAccess  (expected to hang here)")
    e = HIP.hipMemSetAccess(dptr, ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1))
    _mark(q, f"<- hipMemSetAccess ret={e}")

    HIP.hipMemUnmap(dptr, ctypes.c_size_t(size))
    HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
    HIP.hipMemRelease(gen_handle)
    q.put(("result", "OK" if e == 0 else f"FAIL({e}:{_errstr(HIP, e)})"))


def run_guarded(target, timeout):
    """Run target(q) in a spawned child; return (result_or_None, steps, timed_out)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    p.join(timeout)

    steps, result = [], None
    while True:
        try:
            kind, val = q.get_nowait()
        except Exception:
            break
        if kind == "step":
            steps.append(val)
        elif kind == "result":
            result = val

    timed_out = p.is_alive()
    if timed_out:
        p.terminate()
        p.join()
    return result, steps, timed_out


def _info():
    HIP = load_hip()
    if HIP is None:
        return None, "?", "?"
    _bind(HIP)
    v = ctypes.c_int(0)
    try:
        HIP.hipRuntimeGetVersion(ctypes.byref(v))
        ver = v.value
    except Exception:
        ver = "?"
    buf = ctypes.create_string_buffer(256)
    try:
        HIP.hipDeviceGetName(buf, 256, ctypes.c_int(0))
        name = buf.value.decode(errors="ignore")
    except Exception:
        name = "?"
    return HIP, ver, name


def main():
    HIP, ver, name = _info()
    if HIP is None:
        print("FATAL: cannot load libamdhip64.so (searched loader path + wheel dirs)", flush=True)
        sys.exit(4)

    print("=" * 68, flush=True)
    print(f"HIP runtime version = {ver}   device0 = {name}", flush=True)
    print(f"hipMemSetAccess hang probe  (timeout = {TEST_TIMEOUT_S}s)", flush=True)
    print("=" * 68, flush=True)

    result, steps, timed_out = run_guarded(_vmm_setaccess_child, TEST_TIMEOUT_S)
    for s in steps:
        print(f"  {s}", flush=True)
    print("=" * 68, flush=True)

    if timed_out:
        last = steps[-1] if steps else "?"
        stuck = "hipMemSetAccess" if "hipMemSetAccess" in last else last
        print(f"[RESULT] HANG — no return after {TEST_TIMEOUT_S}s at: {stuck}", flush=True)
        if "hipMemSetAccess" in last:
            print("VERDICT: BUG REPRODUCED — hipMemSetAccess hangs on VMM memory.", flush=True)
            sys.exit(2)
        print(f"VERDICT: INCONCLUSIVE — a different VMM call hung ({stuck}).", flush=True)
        sys.exit(3)

    print(f"[RESULT] {result}", flush=True)
    if result == "OK":
        print("VERDICT: NO BUG — hipMemSetAccess returned normally on this ROCm.", flush=True)
        sys.exit(0)
    if result and result.startswith("FAIL"):
        print("VERDICT: NO HANG, but hipMemSetAccess returned an error (see above).", flush=True)
        sys.exit(0)
    print(f"VERDICT: INCONCLUSIVE ({result}).", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
