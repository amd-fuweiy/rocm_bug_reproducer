#!/usr/bin/env python3
"""Minimal customer reproducer: MI308X TF32 GEMM has a systematic scale bias."""

import math
import sys

import torch


M, N, K = 320, 1024, 128
SEEDS = range(1234, 1239)
DEPTH = 20
CHAIN_WIDTH = 1024
CHAIN_BATCH = 320
SNAPSHOT_LAYERS = (1, 5, 10, 20)


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def metrics(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, float]:
    output = output.detach().cpu().double().flatten()
    reference = reference.double().flatten()
    difference = output - reference
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(reference)
    ).item()
    max_abs = difference.abs().max().item()
    scale_minus_one = (
        torch.dot(output, reference) / torch.dot(reference, reference)
    ).item() - 1.0
    return relative_l2, max_abs, scale_minus_one


def make_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    # Representative Linear input: non-negative activation and signed weight.
    activation = torch.randn((M, K), generator=generator).abs()
    weight = torch.randn((K, N), generator=generator)
    return activation, weight


def run(seed: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    activation, weight = make_inputs(seed)
    cpu_fp64 = activation.double() @ weight.double()
    activation_gpu = activation.cuda()
    weight_gpu = weight.cuda()

    set_tf32(False)
    gpu_fp32 = activation_gpu @ weight_gpu
    torch.cuda.synchronize()

    set_tf32(True)
    gpu_tf32 = activation_gpu @ weight_gpu
    torch.cuda.synchronize()
    set_tf32(False)

    return metrics(gpu_fp32, cpu_fp64), metrics(gpu_tf32, cpu_fp64)


def run_depth_chain() -> tuple[
    dict[int, tuple[float, float, float]],
    dict[int, tuple[float, float, float]],
]:
    generator = torch.Generator().manual_seed(1234)
    activation = torch.randn(
        (CHAIN_BATCH, CHAIN_WIDTH), generator=generator
    ).abs()
    weights = [
        torch.randn(
            (CHAIN_WIDTH, CHAIN_WIDTH), generator=generator
        ) * math.sqrt(2.0 / CHAIN_WIDTH)
        for _ in range(DEPTH)
    ]

    references = {}
    cpu_state = activation.double()
    for layer, weight in enumerate(weights, start=1):
        cpu_state = torch.relu(cpu_state @ weight.double().T)
        if layer in SNAPSHOT_LAYERS:
            references[layer] = cpu_state.clone()

    def run_mode(tf32_enabled: bool) -> dict[int, tuple[float, float, float]]:
        set_tf32(tf32_enabled)
        gpu_state = activation.cuda()
        snapshots = {}
        for layer, weight in enumerate(weights, start=1):
            gpu_state = torch.relu(gpu_state @ weight.cuda().T)
            if layer in SNAPSHOT_LAYERS:
                snapshots[layer] = metrics(gpu_state, references[layer])
        torch.cuda.synchronize()
        set_tf32(False)
        return snapshots

    return run_mode(False), run_mode(True)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no ROCm GPU is visible", file=sys.stderr)
        return 2

    properties = torch.cuda.get_device_properties(0)
    print(f"device : {properties.name} ({getattr(properties, 'gcnArchName', '')})")
    print(f"torch  : {torch.__version__}")
    print(f"HIP    : {torch.version.hip}")
    print(f"shape  : (M,N,K)=({M},{N},{K})")
    print()
    print(
        f"{'seed':>6} {'mode':>10} {'relative L2':>14} "
        f"{'max abs':>12} {'scale - 1':>12}"
    )

    fp32_results = []
    tf32_results = []
    for seed in SEEDS:
        fp32, tf32 = run(seed)
        fp32_results.append(fp32)
        tf32_results.append(tf32)
        print(
            f"{seed:>6} {'FP32':>10} {fp32[0] * 100:>13.6f}% "
            f"{fp32[1]:>12.4e} {fp32[2]:>+12.4e}"
        )
        print(
            f"{seed:>6} {'TF32':>10} {tf32[0] * 100:>13.6f}% "
            f"{tf32[1]:>12.4e} {tf32[2]:>+12.4e}"
        )

    mean_fp32_l2 = sum(result[0] for result in fp32_results) / len(fp32_results)
    mean_tf32_l2 = sum(result[0] for result in tf32_results) / len(tf32_results)
    mean_tf32_scale = sum(result[2] for result in tf32_results) / len(tf32_results)

    print()
    print(f"mean FP32 relative L2 : {mean_fp32_l2 * 100:.6f}%")
    print(f"mean TF32 relative L2 : {mean_tf32_l2 * 100:.6f}%")
    print(f"mean TF32 scale - 1   : {mean_tf32_scale:+.6e}")

    single_gemm_reproduced = (
        mean_fp32_l2 < 1.0e-5
        and mean_tf32_l2 > 5.0e-4
        and mean_tf32_scale < -5.0e-4
        and all(result[2] < 0.0 for result in tf32_results)
    )

    print()
    print(
        f"=== cumulative error: {DEPTH}-layer "
        f"Linear+ReLU chain, width={CHAIN_WIDTH} ==="
    )
    fp32_depth, tf32_depth = run_depth_chain()
    print(
        f"{'layer':>6} {'FP32 relative L2':>18} {'FP32 scale - 1':>16} "
        f"{'TF32 relative L2':>18} {'TF32 scale - 1':>16}"
    )
    for layer in SNAPSHOT_LAYERS:
        fp32 = fp32_depth[layer]
        tf32 = tf32_depth[layer]
        print(
            f"{layer:>6} {fp32[0] * 100:>17.6f}% {fp32[2]:>+16.4e} "
            f"{tf32[0] * 100:>17.6f}% {tf32[2]:>+16.4e}"
        )

    tf32_growth = tf32_depth[DEPTH][0] / tf32_depth[1][0]
    final_tf32 = tf32_depth[DEPTH]
    final_fp32 = fp32_depth[DEPTH]
    print()
    print(f"TF32 relative-L2 growth, layer 1 -> {DEPTH}: {tf32_growth:.2f}x")
    print(f"linear-growth reference                    : {DEPTH:.2f}x")
    print(f"sqrt-growth reference                      : {math.sqrt(DEPTH):.2f}x")

    cumulative_error_reproduced = (
        final_fp32[0] < 1.0e-4
        and final_tf32[0] > 1.0e-2
        and final_tf32[2] < -1.0e-2
        and tf32_growth > DEPTH * 0.7
    )
    reproduced = single_gemm_reproduced and cumulative_error_reproduced
    print()
    print(
        "PASS: reproduced systematic TF32 bias and cumulative multi-layer error"
        if reproduced
        else "FAIL: expected MI308X TF32 behavior was not fully reproduced"
    )
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
