#!/usr/bin/env python3
"""
Pure-HIP, dependency-free probe for the root-cause primitive behind the OpenRLHF colocate
weight-sync failure on ROCm:

    hipIpcGetMemHandle() on VMM (hipMemCreate/hipMemMap) memory  ->  hipErrorInvalidValue
    == "CUDA error: invalid argument" seen in torch _share_cuda_ / _new_shared_cuda.

vLLM sleep mode (CuMemAllocator) allocates weights with the HIP Virtual Memory Management
(VMM) driver API (hipMemCreate + hipMemMap), NOT hipMalloc. OpenRLHF colocate then tries to
export those weights over CUDA/HIP IPC (hipIpcGetMemHandle) to the co-located vLLM engine.
HIP IPC only supports hipMalloc'd memory, so it fails.

This probe uses ONLY ctypes + libamdhip64 (no torch, no vLLM, no ray, no openrlhf), so it runs
on ANY ROCm image and is ideal for cross-version regression testing.

It runs two sub-tests on the current GPU and prints PASS/FAIL for each:
  [T1] hipMalloc  -> hipIpcGetMemHandle   (expected: OK on every ROCm)
  [T2] VMM alloc  -> hipIpcGetMemHandle   (the bug: hipErrorInvalidValue on affected ROCm)

Each sub-test runs in its own spawned child process guarded by a hard timeout, so a HIP call
that hangs forever (observed for hipMemSetAccess on some ROCm 7.14 builds) is reported as
HANG(<call>) instead of freezing the whole probe.

Overall exit code:
  0  -> T1 ok AND T2 ok            (no bug: VMM memory is IPC-exportable)
  2  -> T1 ok AND T2 == invalid    (BUG reproduced: VMM memory not IPC-exportable)
  3  -> VMM not even supported / other setup error (inconclusive)
  4  -> T1 failed (environment problem, inconclusive)
"""
import ctypes
import glob
import multiprocessing as mp
import os
import sys

# How long (seconds) any single sub-test may run before we declare a HANG.
TEST_TIMEOUT_S = 30

SIZE = 2 * 1024 * 1024  # 2 MiB (VMM granularity is typically 2 MiB)

# ---- HIP enums / constants ----
hipMemAllocationTypePinned = 1
hipMemLocationTypeDevice = 1
hipMemAccessFlagsProtReadWrite = 3
hipMemAllocationGranularityMinimum = 0
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


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _bind(HIP):
    for fn in ("hipMalloc", "hipFree", "hipIpcGetMemHandle", "hipMemGetAllocationGranularity",
               "hipMemCreate", "hipMemAddressReserve", "hipMemMap", "hipMemSetAccess",
               "hipMemUnmap", "hipMemAddressFree", "hipMemRelease", "hipRuntimeGetVersion",
               "hipDeviceGetName", "hipSetDevice", "hipGetErrorString"):
        if hasattr(HIP, fn):
            getattr(HIP, fn).restype = ctypes.c_int
    if hasattr(HIP, "hipGetErrorString"):
        HIP.hipGetErrorString.restype = ctypes.c_char_p


def _errstr(HIP, err):
    try:
        return HIP.hipGetErrorString(ctypes.c_int(err)).decode()
    except Exception:
        return "?"


def _mark(q, step):
    """Report progress so the parent knows which call was in flight if we hang."""
    if q is not None:
        q.put(("step", step))


# --------------------------------------------------------------------------------------
# Sub-tests. Each is a top-level function so it can run in a spawned child process.
# They push ("step", name) markers then a final ("result", value) onto the queue.
# --------------------------------------------------------------------------------------
def _t1_child(q):
    HIP = load_hip()
    if HIP is None:
        q.put(("result", "SETUP_FAIL(no libamdhip64)"))
        return
    _bind(HIP)
    HIP.hipSetDevice(ctypes.c_int(0))
    ptr = ctypes.c_void_p()
    _mark(q, "hipMalloc")
    if HIP.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(SIZE)) != 0:
        q.put(("result", "SETUP_FAIL(hipMalloc)"))
        return
    handle = hipIpcMemHandle_t()
    _mark(q, "hipIpcGetMemHandle")
    err = HIP.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    HIP.hipFree(ptr)
    q.put(("result", "OK" if err == 0 else f"FAIL({err}:{_errstr(HIP, err)})"))


def _t2_child(q):
    HIP = load_hip()
    if HIP is None:
        q.put(("result", "VMM_SETUP_FAIL(no libamdhip64)"))
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
    _mark(q, "hipMemGetAllocationGranularity")
    HIP.hipMemGetAllocationGranularity(
        ctypes.byref(granularity), ctypes.byref(prop),
        ctypes.c_int(hipMemAllocationGranularityRecommended),
    )
    gran = granularity.value or (2 * 1024 * 1024)
    size = ((SIZE + gran - 1) // gran) * gran

    gen_handle = ctypes.c_ulonglong(0)
    _mark(q, "hipMemCreate")
    e = HIP.hipMemCreate(ctypes.byref(gen_handle), ctypes.c_size_t(size), ctypes.byref(prop), ctypes.c_ulonglong(0))
    if e != 0:
        q.put(("result", f"VMM_SETUP_FAIL(hipMemCreate={e}:{_errstr(HIP, e)})"))
        return

    dptr = ctypes.c_void_p()
    _mark(q, "hipMemAddressReserve")
    e = HIP.hipMemAddressReserve(ctypes.byref(dptr), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_ulonglong(0))
    if e != 0:
        HIP.hipMemRelease(gen_handle)
        q.put(("result", f"VMM_SETUP_FAIL(AddressReserve={e}:{_errstr(HIP, e)})"))
        return

    _mark(q, "hipMemMap")
    e = HIP.hipMemMap(dptr, ctypes.c_size_t(size), ctypes.c_size_t(0), gen_handle, ctypes.c_ulonglong(0))
    if e != 0:
        HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
        HIP.hipMemRelease(gen_handle)
        q.put(("result", f"VMM_SETUP_FAIL(hipMemMap={e}:{_errstr(HIP, e)})"))
        return

    # The actual probe: export an IPC handle from VMM memory. hipMemSetAccess is deliberately
    # NOT called here -- HIP IPC rejects VMM handles based on the allocation *type*, not on
    # access flags, and hipMemSetAccess is known to hang on some ROCm 7.14 builds (see _t2b).
    handle = hipIpcMemHandle_t()
    _mark(q, "hipIpcGetMemHandle")
    err = HIP.hipIpcGetMemHandle(ctypes.byref(handle), dptr)

    HIP.hipMemUnmap(dptr, ctypes.c_size_t(size))
    HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
    HIP.hipMemRelease(gen_handle)

    if err == 0:
        q.put(("result", "OK"))
    elif err == 1:  # hipErrorInvalidValue == CUDA "invalid argument"
        q.put(("result", f"INVALID_ARGUMENT({err})"))
    else:
        q.put(("result", f"FAIL({err}:{_errstr(HIP, err)})"))


def _t2b_child(q):
    """Diagnostic: does hipMemSetAccess itself work on VMM memory? Observed to hang on
    some ROCm 7.14 builds, which is why the primary T2 probe above skips it."""
    HIP = load_hip()
    if HIP is None or not hasattr(HIP, "hipMemCreate"):
        q.put(("result", "SKIP"))
        return
    _bind(HIP)
    HIP.hipSetDevice(ctypes.c_int(0))

    prop = hipMemAllocationProp()
    prop.type = hipMemAllocationTypePinned
    prop.requestedHandleType = hipMemHandleTypeNone
    prop.location.type = hipMemLocationTypeDevice
    prop.location.id = 0

    granularity = ctypes.c_size_t(0)
    HIP.hipMemGetAllocationGranularity(
        ctypes.byref(granularity), ctypes.byref(prop),
        ctypes.c_int(hipMemAllocationGranularityRecommended),
    )
    gran = granularity.value or (2 * 1024 * 1024)
    size = ((SIZE + gran - 1) // gran) * gran

    gen_handle = ctypes.c_ulonglong(0)
    if HIP.hipMemCreate(ctypes.byref(gen_handle), ctypes.c_size_t(size), ctypes.byref(prop), ctypes.c_ulonglong(0)) != 0:
        q.put(("result", "SETUP_FAIL(hipMemCreate)"))
        return
    dptr = ctypes.c_void_p()
    if HIP.hipMemAddressReserve(ctypes.byref(dptr), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_ulonglong(0)) != 0:
        HIP.hipMemRelease(gen_handle)
        q.put(("result", "SETUP_FAIL(AddressReserve)"))
        return
    if HIP.hipMemMap(dptr, ctypes.c_size_t(size), ctypes.c_size_t(0), gen_handle, ctypes.c_ulonglong(0)) != 0:
        HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
        HIP.hipMemRelease(gen_handle)
        q.put(("result", "SETUP_FAIL(hipMemMap)"))
        return

    desc = hipMemAccessDesc()
    desc.location.type = hipMemLocationTypeDevice
    desc.location.id = 0
    desc.flags = hipMemAccessFlagsProtReadWrite
    _mark(q, "hipMemSetAccess")
    e = HIP.hipMemSetAccess(dptr, ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1))

    HIP.hipMemUnmap(dptr, ctypes.c_size_t(size))
    HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
    HIP.hipMemRelease(gen_handle)
    q.put(("result", "OK" if e == 0 else f"FAIL({e}:{_errstr(HIP, e)})"))


def run_guarded(target, timeout):
    """Run target(q) in a spawned child; return its result string, or HANG(<last step>)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    p.join(timeout)

    last_step = "?"
    result = None
    while True:
        try:
            kind, val = q.get_nowait()
        except Exception:
            break
        if kind == "step":
            last_step = val
        elif kind == "result":
            result = val

    if p.is_alive():
        p.terminate()
        p.join()
        return f"HANG({last_step})"
    if result is not None:
        return result
    # child exited without a result and without hanging -> crashed
    return f"CHILD_CRASH(after {last_step}, exit={p.exitcode})"


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
    print("=" * 68, flush=True)

    t1 = run_guarded(_t1_child, TEST_TIMEOUT_S)
    print(f"[T1] hipMalloc -> hipIpcGetMemHandle : {t1}", flush=True)
    t2 = run_guarded(_t2_child, TEST_TIMEOUT_S)
    print(f"[T2] VMM alloc -> hipIpcGetMemHandle : {t2}", flush=True)
    t2b = run_guarded(_t2b_child, TEST_TIMEOUT_S)
    print(f"[T2*] VMM alloc -> hipMemSetAccess   : {t2b}  (diagnostic)", flush=True)
    print("=" * 68, flush=True)

    if t1 != "OK":
        print("VERDICT: INCONCLUSIVE (baseline hipMalloc IPC failed).", flush=True)
        sys.exit(4)
    if t2 == "OK":
        print("VERDICT: NO BUG — VMM memory IS IPC-exportable on this ROCm.", flush=True)
        sys.exit(0)
    if t2.startswith("INVALID_ARGUMENT"):
        print("VERDICT: BUG REPRODUCED — VMM memory is NOT IPC-exportable "
              "(hipIpcGetMemHandle -> invalid argument).", flush=True)
        sys.exit(2)
    if t2.startswith("HANG"):
        print(f"VERDICT: INCONCLUSIVE (a VMM HIP call hung: {t2}).", flush=True)
        sys.exit(3)
    print(f"VERDICT: INCONCLUSIVE (VMM setup issue: {t2}).", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
