import ctypes
import multiprocessing as mp

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
    hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return (total.value - free.value) / (1024**3)

class HipIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]

def _worker_loop(recv_conn, ack_conn):
    hip, _ = load_hip()
    hip_ptr_t = ctypes.c_void_p
    hip.hipSetDevice.argtypes = [ctypes.c_int]
    hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(hip_ptr_t), HipIpcMemHandle, ctypes.c_uint]
    hip.hipIpcCloseMemHandle.argtypes = [hip_ptr_t]
    
    # 子进程也需要绑定 Device
    hip.hipSetDevice(0)

    while True:
        msg = recv_conn.recv()
        if msg == b"STOP":
            ack_conn.send("ok")
            break
        
        handle = HipIpcMemHandle.from_buffer_copy(msg)
        ptr = hip_ptr_t()
        
        # flags 必须为 hipIpcMemLazyEnablePeerAccess (即 1)
        err = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), handle, 1)
        if err != 0:
            ack_conn.send(f"hipIpcOpenMemHandle failed: {err}")
            continue
        
        err = hip.hipIpcCloseMemHandle(ptr)
        if err != 0:
            ack_conn.send(f"hipIpcCloseMemHandle failed: {err}")
            continue
            
        ack_conn.send("ok")

def run_mp_baseline(iterations=10, alloc_mb=500):
    hip, _ = load_hip()
    hip_ptr_t = ctypes.c_void_p
    hip.hipSetDevice.argtypes = [ctypes.c_int]
    hip.hipMalloc.argtypes = [ctypes.POINTER(hip_ptr_t), ctypes.c_size_t]
    hip.hipFree.argtypes = [hip_ptr_t]
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(HipIpcMemHandle), hip_ptr_t]
    
    hip.hipSetDevice(0)

    mp.set_start_method("spawn", force=True)
    parent_conn, child_conn = mp.Pipe()
    ack_parent, ack_child = mp.Pipe()
    
    p = mp.Process(target=_worker_loop, args=(child_conn, ack_child))
    p.start()

    start = hip_used_gb(hip)
    print(f"[Pure HIP IPC] start used: {start:.2f} GB")
    
    for i in range(iterations):
        ptr = hip_ptr_t()
        hip.hipMalloc(ctypes.byref(ptr), alloc_mb * 1024 * 1024)
            
        handle = HipIpcMemHandle()
        hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
            
        # 1. 发送 handle 给子进程
        parent_conn.send(bytes(handle))
        
        # 2. 等待子进程 Open 和 Close 完成
        status = ack_parent.recv()
        if status != "ok":
            raise RuntimeError(f"Worker failed: {status}")
            
        # 3. 父进程释放内存
        hip.hipFree(ptr)
            
        now = hip_used_gb(hip)
        print(f"[Pure HIP IPC] step {i:02d}: hip used={now:.2f} GB")

    parent_conn.send(b"STOP")
    ack_parent.recv()
    p.join()
    
    end = hip_used_gb(hip)
    growth = end - start
    print(f"[Pure HIP IPC] used: {start:.2f} -> {end:.2f} GB (growth={growth:.2f} GB)")

if __name__ == "__main__":
    run_mp_baseline()