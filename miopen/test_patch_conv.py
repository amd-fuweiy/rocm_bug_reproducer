import json
import statistics
import torch
import torch.nn.functional as F
assert torch.version.hip, torch.__version__
assert torch.cuda.is_available()
props = torch.cuda.get_device_properties(0)
arch = props.gcnArchName.split(":", 1)[0]
assert arch == "gfx942", props.gcnArchName
print("ENV", json.dumps({
   "torch": torch.__version__,
   "hip": torch.version.hip,
   "device": props.name,
   "arch": props.gcnArchName,
}))
torch.manual_seed(123)
device = "cuda"
dtype = torch.bfloat16
# SeedVL 真实结构：每 rank 约 9500 个独立 14x14 patch。
N, C, P, D = 9500, 3, 14, 1280
x = torch.randn(N, C, P, P, device=device, dtype=dtype)
conv = torch.nn.Conv2d(
   C, D, kernel_size=P, stride=P, padding=0,
   bias=True, device=device, dtype=dtype,
)
# 数学等价的 GEMM control；复用完全相同的参数。
linear = torch.nn.Linear(C * P * P, D, bias=True,
                        device=device, dtype=dtype)
with torch.no_grad():
   linear.weight.copy_(conv.weight.flatten(1))
   linear.bias.copy_(conv.bias)
@torch.no_grad()
def parity():
   y_conv = conv(x).flatten(1)
   y_linear = linear(x.flatten(1))
   delta = y_conv.float() - y_linear.float()
   rel_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(y_conv.float())
   cosine = F.cosine_similarity(
       y_conv.float().flatten(), y_linear.float().flatten(), dim=0
   )
   print("PARITY", json.dumps({
       "max_abs": delta.abs().max().item(),
       "mean_abs": delta.abs().mean().item(),
       "rel_l2": rel_l2.item(),
       "cosine": cosine.item(),
       "bitwise": torch.equal(y_conv, y_linear),
   }))

def measure(fn, warmup=5, iterations=15):
   for _ in range(warmup):
       fn()
   torch.cuda.synchronize()
   samples = []
   for _ in range(iterations):
       begin = torch.cuda.Event(enable_timing=True)
       end = torch.cuda.Event(enable_timing=True)
       begin.record()
       fn()
       end.record()
       end.synchronize()
       samples.append(begin.elapsed_time(end))
   return {
       "median_ms": statistics.median(samples),
       "min_ms": min(samples),
       "max_ms": max(samples),
   }

def conv_forward():
   return conv(x)

def linear_forward():
   return linear(x.flatten(1))

def conv_forward_backward():
   conv.zero_grad(set_to_none=True)
   z = x.detach().requires_grad_(True)
   loss = conv(z).float().square().mean()
   loss.backward()

def linear_forward_backward():
   linear.zero_grad(set_to_none=True)
   z = x.detach().requires_grad_(True)
   loss = linear(z.flatten(1)).float().square().mean()
   loss.backward()

parity()
result = {
   "shape": [N, C, P, P],
   "conv": {
       "forward": measure(conv_forward),
       "forward_backward": measure(conv_forward_backward, warmup=3, iterations=7),
   },
   "linear_control": {
       "forward": measure(linear_forward),
       "forward_backward": measure(linear_forward_backward, warmup=3, iterations=7),
   },
}
result["speedup"] = {
   "forward": result["conv"]["forward"]["median_ms"] /
              result["linear_control"]["forward"]["median_ms"],
   "forward_backward": result["conv"]["forward_backward"]["median_ms"] /
                       result["linear_control"]["forward_backward"]["median_ms"],
}
print("RESULT", json.dumps(result, indent=2))
