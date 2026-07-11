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

Overall exit code:
  0  -> T1 ok AND T2 ok            (no bug: VMM memory is IPC-exportable)
  2  -> T1 ok AND T2 == invalid    (BUG reproduced: VMM memory not IPC-exportable)
  3  -> VMM not even supported / other setup error (inconclusive)
  4  -> T1 failed (environment problem, inconclusive)
"""
import ctypes
import sys

HIP = None
for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5"):
    try:
        HIP = ctypes.CDLL(name)
        break
    except OSError:
        continue
if HIP is None:
    print("FATAL: cannot load libamdhip64.so", flush=True)
    sys.exit(4)

# ---- HIP enums / constants ----
hipMemAllocationTypePinned = 1
hipMemLocationTypeDevice = 1
hipMemAccessFlagsProtReadWrite = 3
hipMemAllocationGranularityMinimum = 0
hipMemHandleTypeNone = 0

SIZE = 2 * 1024 * 1024  # 2 MiB (VMM granularity is typically 2 MiB)


def chk(name, err, allow=()):
    if err == 0 or err in allow:
        return err
    # get error string
    HIP.hipGetErrorString.restype = ctypes.c_char_p
    msg = HIP.hipGetErrorString(ctypes.c_int(err)).decode()
    print(f"  {name} -> hipError {err} ({msg})", flush=True)
    return err


class hipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class hipMemAllocationProp(ctypes.Structure):
    # struct hipMemAllocationProp { type; requestedHandleType; location; win32HandleMetaData;
    #   struct { unsigned char compressionType; unsigned char gpuDirectRDMACapable;
    #            unsigned short usage; unsigned char reserved[4]; } allocFlags; }
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


def hip_version():
    v = ctypes.c_int(0)
    try:
        HIP.hipRuntimeGetVersion(ctypes.byref(v))
    except Exception:
        return "?"
    return v.value


def dev_name(dev=0):
    buf = ctypes.create_string_buffer(256)
    try:
        HIP.hipDeviceGetName(buf, 256, ctypes.c_int(dev))
        return buf.value.decode(errors="ignore")
    except Exception:
        return "?"


def t1_hipmalloc_ipc():
    ptr = ctypes.c_void_p()
    if chk("hipMalloc", HIP.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(SIZE))) != 0:
        return "SETUP_FAIL"
    handle = hipIpcMemHandle_t()
    err = HIP.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    chk("hipIpcGetMemHandle(hipMalloc)", err)
    HIP.hipFree(ptr)
    return "OK" if err == 0 else f"FAIL({err})"


def t2_vmm_ipc():
    # 1) VMM supported?
    supported = ctypes.c_int(0)
    # hipDeviceAttributeVirtualMemoryManagementSupported — value varies; query defensively.
    # We instead just try to create; if hipMemCreate is missing/unsupported we report SETUP.
    if not hasattr(HIP, "hipMemCreate"):
        return "VMM_UNSUPPORTED(no hipMemCreate)"

    prop = hipMemAllocationProp()
    prop.type = hipMemAllocationTypePinned
    prop.requestedHandleType = hipMemHandleTypeNone
    prop.location.type = hipMemLocationTypeDevice
    prop.location.id = 0

    granularity = ctypes.c_size_t(0)
    HIP.hipMemGetAllocationGranularity(
        ctypes.byref(granularity), ctypes.byref(prop), ctypes.c_int(hipMemAllocationGranularityMinimum)
    )
    gran = granularity.value or (2 * 1024 * 1024)
    size = ((SIZE + gran - 1) // gran) * gran

    gen_handle = ctypes.c_ulonglong(0)
    e = chk("hipMemCreate", HIP.hipMemCreate(ctypes.byref(gen_handle), ctypes.c_size_t(size), ctypes.byref(prop), ctypes.c_ulonglong(0)))
    if e != 0:
        return f"VMM_SETUP_FAIL(hipMemCreate={e})"

    dptr = ctypes.c_void_p()
    e = chk("hipMemAddressReserve", HIP.hipMemAddressReserve(ctypes.byref(dptr), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_ulonglong(0)))
    if e != 0:
        HIP.hipMemRelease(gen_handle)
        return f"VMM_SETUP_FAIL(AddressReserve={e})"

    e = chk("hipMemMap", HIP.hipMemMap(dptr, ctypes.c_size_t(size), ctypes.c_size_t(0), gen_handle, ctypes.c_ulonglong(0)))
    if e != 0:
        HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
        HIP.hipMemRelease(gen_handle)
        return f"VMM_SETUP_FAIL(hipMemMap={e})"

    desc = hipMemAccessDesc()
    desc.location.type = hipMemLocationTypeDevice
    desc.location.id = 0
    desc.flags = hipMemAccessFlagsProtReadWrite
    HIP.hipMemSetAccess(dptr, ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1))

    # 2) The actual probe: export an IPC handle from VMM memory.
    handle = hipIpcMemHandle_t()
    err = HIP.hipIpcGetMemHandle(ctypes.byref(handle), dptr)
    chk("hipIpcGetMemHandle(VMM)", err)

    # cleanup
    HIP.hipMemUnmap(dptr, ctypes.c_size_t(size))
    HIP.hipMemAddressFree(dptr, ctypes.c_size_t(size))
    HIP.hipMemRelease(gen_handle)

    if err == 0:
        return "OK"
    # hipErrorInvalidValue == 1 (== CUDA "invalid argument")
    return f"INVALID_ARGUMENT({err})" if err == 1 else f"FAIL({err})"


def main():
    # bind restypes
    for fn in ("hipMalloc", "hipFree", "hipIpcGetMemHandle", "hipMemGetAllocationGranularity",
               "hipMemCreate", "hipMemAddressReserve", "hipMemMap", "hipMemSetAccess",
               "hipMemUnmap", "hipMemAddressFree", "hipMemRelease", "hipRuntimeGetVersion",
               "hipDeviceGetName", "hipSetDevice"):
        if hasattr(HIP, fn):
            getattr(HIP, fn).restype = ctypes.c_int
    HIP.hipSetDevice(ctypes.c_int(0))

    print("=" * 68, flush=True)
    print(f"HIP runtime version = {hip_version()}   device0 = {dev_name(0)}", flush=True)
    print("=" * 68, flush=True)

    t1 = t1_hipmalloc_ipc()
    print(f"[T1] hipMalloc -> hipIpcGetMemHandle : {t1}", flush=True)
    t2 = t2_vmm_ipc()
    print(f"[T2] VMM alloc -> hipIpcGetMemHandle : {t2}", flush=True)
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
    print(f"VERDICT: INCONCLUSIVE (VMM setup issue: {t2}).", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
