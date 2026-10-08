#!/usr/bin/env python3
"""One GPU, identical masks. Fails only when the tensor lives in an expandable segment."""
import os, pickle, subprocess, sys, tempfile, time

HERE = os.path.abspath(__file__)

def producer(path):
    import torch
    torch.cuda.set_device(0)
    t = torch.arange(4096, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    from torch.multiprocessing.reductions import reduce_tensor
    _, args = reduce_tensor(t)
    pickle.dump({"args": args, "checksum": int(t.sum().item())}, open(path, "wb"))
    print("PRODUCER wrote handle", flush=True)
    time.sleep(30)

def consumer(path):
    import torch
    while not os.path.exists(path):
        time.sleep(0.05)
    payload = pickle.load(open(path, "rb"))
    torch.cuda.set_device(0)
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    t = rebuild_cuda_tensor(*payload["args"])
    got = int(t.sum().item())
    print("CHECKSUM_OK" if got == payload["checksum"] else f"CHECKSUM_BAD {got}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("producer", "consumer"):
        (producer if sys.argv[1] == "producer" else consumer)(sys.argv[2])
        raise SystemExit(0)
    fd, path = tempfile.mkstemp(prefix="ipc-")
    os.close(fd)
    os.unlink(path)
    env = os.environ.copy()
    prod = subprocess.Popen([sys.executable, HERE, "producer", path], env=env)
    cons = subprocess.Popen([sys.executable, HERE, "consumer", path], env=env)
    rc = cons.wait(timeout=60)
    prod.terminate()
    raise SystemExit(rc)
