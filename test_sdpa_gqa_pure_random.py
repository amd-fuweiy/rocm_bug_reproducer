"""
Pure random reproduction of SDPA enable_gqa=True bug on PyTorch 2.8.0+rocm7.0.2.

Root cause: The GQA SDPA kernel mishandles non-contiguous V tensors when Q/K
are contiguous. This matches real model behavior where Q/K become contiguous
after RoPE (apply_rotary_pos_emb), but V stays non-contiguous from transpose.

Trigger condition:
  Q: contiguous     [B, Q_H, S, D]
  K: contiguous     [B, KV_H, S, D]
  V: NON-contiguous [B, KV_H, S, D]  (from .transpose(1, 2))

This test uses ONLY random tensors — no model weights needed.

Workarounds:
  1. v = v.contiguous() before SDPA
  2. --attn_implementation eager
  3. Upgrade to ROCm 7.2+
"""
import sys
import torch
import torch.nn.functional as F


def check(q, k, v, label, scale):
    ng = q.shape[1] // k.shape[1]
    ke = k.repeat_interleave(ng, dim=1)
    ve = v.repeat_interleave(ng, dim=1)
    try:
        out_ref = F.scaled_dot_product_attention(q, ke, ve, is_causal=True, scale=scale)
        out_gqa = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
        torch.cuda.synchronize()
        d = (out_gqa.float() - out_ref.float()).abs().max().item()
    except RuntimeError as e:
        print(f"  {label:55s} CRASH: {e}")
        return True  # crash = bug
    bug = d > 1.0
    tag = "FAIL" if bug else "PASS"
    print(f"  {label:55s} diff={d:8.4f} [{tag}]")
    return bug


def main():
    device = torch.device("cuda:0")
    print(f"torch: {torch.__version__}")
    print(f"hip:   {torch.version.hip}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}\n")

    B, S, Q_H, KV_H, D = 1, 512, 8, 2, 256
    scale = 1.0 / (D ** 0.5)
    any_bug = False

    # ── Test 1: V non-contiguous (transpose), Q/K contiguous ──
    print("=" * 80)
    print("Test 1: Q(contiguous) K(contiguous) V(non-contiguous from transpose)")
    print("        This matches real GQA model layout (Q/K contiguous after RoPE)")
    print("=" * 80)
    for trial in range(5):
        torch.manual_seed(trial)
        q = torch.randn(B, Q_H, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, KV_H, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, S, KV_H, D, device=device, dtype=torch.bfloat16).transpose(1, 2)
        assert q.is_contiguous() and k.is_contiguous() and not v.is_contiguous()
        if check(q, k, v, f"seed={trial}", scale):
            any_bug = True

    # ── Test 2: Control — all contiguous ──
    print()
    print("=" * 80)
    print("Test 2: Control — all contiguous (V.contiguous() workaround)")
    print("=" * 80)
    for trial in range(3):
        torch.manual_seed(trial)
        q = torch.randn(B, Q_H, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, KV_H, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, S, KV_H, D, device=device, dtype=torch.bfloat16).transpose(1, 2).contiguous()
        check(q, k, v, f"seed={trial} (V made contiguous)", scale)

    # ── Test 3: Control — all non-contiguous ──
    print()
    print("=" * 80)
    print("Test 3: Control — all non-contiguous")
    print("=" * 80)
    for trial in range(3):
        torch.manual_seed(trial)
        q = torch.randn(B, S, Q_H, D, device=device, dtype=torch.bfloat16).transpose(1, 2)
        k = torch.randn(B, S, KV_H, D, device=device, dtype=torch.bfloat16).transpose(1, 2)
        v = torch.randn(B, S, KV_H, D, device=device, dtype=torch.bfloat16).transpose(1, 2)
        check(q, k, v, f"seed={trial} (all transposed)", scale)

    # ── Test 4: Multiple GQA shapes ──
    print()
    print("=" * 80)
    print("Test 4: Multiple shapes — V non-contiguous, Q/K contiguous")
    print("=" * 80)
    configs = [
        (1, 128, 8, 2, 128),
        (1, 256, 8, 2, 256),
        (1, 512, 8, 2, 256),
        (1, 1024, 8, 2, 256),
        (1, 512, 16, 4, 128),
        (1, 512, 32, 8, 128),
        (1, 512, 4, 1, 256),
        (2, 512, 8, 2, 256),
    ]
    for b, s, qh, kvh, d in configs:
        torch.manual_seed(42)
        sc = 1.0 / (d ** 0.5)
        q = torch.randn(b, qh, s, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, kvh, s, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, s, kvh, d, device=device, dtype=torch.bfloat16).transpose(1, 2)
        if check(q, k, v, f"B={b} S={s:4d} Q={qh:2d} KV={kvh} D={d:3d}", sc):
            any_bug = True

    # ── Test 5: Backward ──
    print()
    print("=" * 80)
    print("Test 5: Backward correctness — V non-contiguous")
    print("=" * 80)
    bwd_bug = False
    for trial in range(3):
        torch.manual_seed(trial)

        q1 = torch.randn(B, Q_H, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k1 = torch.randn(B, KV_H, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v1 = torch.randn(B, S, KV_H, D, device=device, dtype=torch.bfloat16).transpose(1, 2).contiguous()
        v1.requires_grad_(True)
        ke1 = k1.repeat_interleave(Q_H // KV_H, dim=1)
        ve1 = v1.repeat_interleave(Q_H // KV_H, dim=1)
        out1 = F.scaled_dot_product_attention(q1, ke1, ve1, is_causal=True, scale=scale)
        out1.sum().backward()

        torch.manual_seed(trial)
        q2 = torch.randn(B, Q_H, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k2 = torch.randn(B, KV_H, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v2_base = torch.randn(B, S, KV_H, D, device=device, dtype=torch.bfloat16)
        v2 = v2_base.transpose(1, 2)
        v2.requires_grad_(True)
        try:
            out2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, scale=scale, enable_gqa=True)
            out2.sum().backward()
            torch.cuda.synchronize()
            fwd_d = (out2.float() - out1.float()).abs().max().item()
            gq_d = (q2.grad.float() - q1.grad.float()).abs().max().item()
            bug = fwd_d > 1.0 or gq_d > 1.0
        except RuntimeError:
            fwd_d = gq_d = float('inf')
            bug = True
        if bug:
            bwd_bug = True
            any_bug = True
        print(f"  seed={trial}: fwd_diff={fwd_d:.4f}  grad_Q_diff={gq_d:.4f}  [{'FAIL' if bug else 'PASS'}]")

    # ── Summary ──
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    if any_bug:
        print("  *** BUG CONFIRMED with pure random tensors ***")
        print("  *** F.scaled_dot_product_attention(enable_gqa=True) produces wrong ***")
        print("  *** results when V is non-contiguous (transposed) on this platform ***")
        print()
        print("  Workarounds:")
        print("    1. v = v.contiguous() before calling SDPA")
        print("    2. Use eager attention: --attn_implementation eager")
        print("    3. Upgrade to ROCm 7.2+")
    else:
        print("  All tests passed. No bug on this platform.")
    return 1 if any_bug else 0


if __name__ == "__main__":
    sys.exit(main())
