import ray
import torch
import io
import gc
from multiprocessing.reduction import ForkingPickler
import time
ray.init(ignore_reinit_error=True)
@ray.remote(num_gpus=0.1)
class Receiver:
    def __init__(self):
        self.device = torch.device("cuda:0")
    def receive_bytes(self, serialized_bytes):
        # Simulate MultiprocessingSerializer.deserialize
        import pickle
        obj = pickle.loads(serialized_bytes)
        # Use the object briefly
        _ = obj[0].item()
        # Explicit cleanup on receiver side
        del obj
        gc.collect()
        torch.cuda.empty_cache()
        return True
def serialize_tensor(tensor):
    buf = io.BytesIO()
    ForkingPickler(buf, -1).dump(tensor)
    return buf.getvalue()
def main():
    receiver = Receiver.remote()
    device = torch.device("cuda:0")
    print("=== Start reproducing VRAM growth (memory leak) issue ===")
    # Record initial GPU memory usage
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    initial_used_gb = (total_mem - free_mem) / (1024**3)
    print(f"Initial: Physical GPU memory usage = {initial_used_gb:.2f} GB\n")
    for i in range(10):
        # 1. Allocate a brand-new tensor every iteration
        #    (~500 MB: 125,000,000 * 4 bytes)
        new_tensor = torch.empty(
            125_000_000, dtype=torch.float32, device=device
        )
        # 2. Serialize tensor into raw bytes using ForkingPickler
        #    (simulating MultiprocessingSerializer)
        serialized_bytes = serialize_tensor(new_tensor)
        # 3. Send bytes asynchronously to Ray actor
        ray.get(receiver.receive_bytes.remote(serialized_bytes))
        # 4. Attempt to release memory on sender side
        del new_tensor
        del serialized_bytes
        gc.collect()
        torch.cuda.empty_cache()
        # 5. Query actual physical GPU memory usage
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        used_mem_gb = (total_mem - free_mem) / (1024**3)
        print(f"Step {i}: Physical GPU memory usage = {used_mem_gb:.2f} GB")
        time.sleep(0.5)
    print("\n=== Conclusion ===")
    print(
        "Even though we call del, gc.collect(), and empty_cache() at every step, "
        "physical GPU memory keeps increasing."
    )
    print(
        "This happens because when ForkingPickler serializes a CUDA tensor, "
        "it registers the tensor in a background CudaIPCCollect mechanism, "
        "holding a reference until an ACK is received from the peer via "
        "PyTorch's native multiprocessing.Queue."
    )
    print(
        "However, in this setup, the serialized bytes are sent through Ray, "
        "so PyTorch never receives the expected ACK signal."
    )
    print(
        "Since a new tensor is allocated every iteration, these large tensors "
        "remain pinned in the IPC cache indefinitely, leading to severe "
        "apparent GPU memory leakage."
    )
if __name__ == "__main__":
    main()
