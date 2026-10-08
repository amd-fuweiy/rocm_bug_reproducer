# RCCL: `TopoPreset: Local rank N not found in intra-node graph` with one GPU per process

## Summary

Since rocm-systems commit `5f38cd0701` (#10640), communicator init fails on a full 8-GPU node when
each process sees only its own GPU through `HIP_VISIBLE_DEVICES`. This is the standard
one-device-per-process layout used by Ray and torchrun-style launchers. RCCL 2.30.4 works. Builds
from `5f38cd0701` onward fail, through develop `4bab2283` (2.31.2). The latest ROCm build,
`rocm/frameworks-internal:rocm10.2.0a20261007_ubuntu24.04_py3.12_pytorch_2.12.0_9ccf2fd`, also fails.

```
rank 1..7: ncclInternalError: Internal check failed.
           TopoPreset: Local rank <r> not found in intra-node graph. Topology misconfiguration.
rank 0:    ncclRemoteError ... socketProgress: Connection closed by remote peer
```

## Trigger conditions

All three conditions must hold:

1. **The communicator spans every GPU of the node**, so a predefined Rome model is matched. On
   8x MI308X, 2 to 7 ranks pass and 8 ranks fail.
2. **Two or more ranks have the same HIP device ordinal.** The typical case is
   `HIP_VISIBLE_DEVICES=<i>` per process, which makes every process's device 0. An own-GPU-first 8-GPU
   mask with ordinal 0 on every rank hits it too, when torch passes `device_id`. Without `device_id`,
   that layout is rejected earlier with `Multiple Ranks are using the same GPU`, on 2.30.4 as well.
   `ROCR_VISIBLE_DEVICES` renumbers the same way,
   but it was not run through the reproducer.
3. **P2P is enabled**, which is the default. `NCCL_P2P_DISABLE=1` avoids the failure.

When every process sees `HIP_VISIBLE_DEVICES=0..7` and binds ordinal = rank, the job passes.

## Root cause

`#10640` changed the `<gpu dev=…>` attribute written by `ncclTopoGetXmlFromGpu`
(`src/graph/xml.cc`). It used to be the SMI index, which is machine-wide. It is now the HIP ordinal
from `hipDeviceGetByPCIBusId`, which is relative to the process's visible-device mask. Every rank
therefore reports `dev=0`, and Rome model matching maps all ring positions to rank 0:

```
good (4c0dc03b09): Found matching Rome model index 38 with GPU mapping: 0 1 2 3 4 5 6 7
bad  (5f38cd0701): Found matching Rome model index 38 with GPU mapping: 0 0 0 0 0 0 0 0
```

`ncclTopoPreset` then cannot find ranks 1-7 in the ring (`src/graph/connect.cc`). The `mlopart=0`
stamping from the same commit was removed later by `97554d81d4`. The `dev` change is still on
develop, and `NCCL_TOPO_SPLIT_MLOPART=0` does not help.

**Confirmation.** On develop `4bab2283`, writing and looking up `dev` by `smiDev` while keeping the
HIP ordinal for `hipGetDeviceProperties` makes every case pass
(`experiment_head_gpu_dev_smi_index.diff`). This is a diagnostic, not a proposed patch. The HIP
ordinal was introduced for CPX/DPX partitions, which this SPX machine cannot test.

## Reproduce

`repro.py` needs only torch. It spawns N processes with `HIP_VISIBLE_DEVICES=<rank>` and runs
`init_process_group("nccl")` followed by one `all_reduce`. It prints the `ncclGetVersion` of the
loaded `librccl` and exits 0 on PASS, 1 on FAIL, and 124 on a hang. Each run takes about 6 seconds.

```bash
# On the host or inside any ROCm torch image, with 8 GPUs:
python3 repro.py --nproc 8                      # FAIL on 5f38cd0701 .. develop
python3 repro.py --nproc 7                      # PASS (no full-node Rome model)
NCCL_P2P_DISABLE=1 python3 repro.py --nproc 8   # PASS

# The same commands in a container; run.sh clears baked HIP/ROCR_VISIBLE_DEVICES:
./run.sh <image> [docker -e/-v args] [-- repro.py args]
```

To test an RCCL build, mount it over the library that torch loads rather than using `LD_PRELOAD`.
torch also loads its own copy, and preloading leaves two RCCLs mapped in one process:

```bash
./run.sh rocm/primus:v26.7 \
  -v $OUT/lib/librccl.so.1.0:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so.1:ro
```

`build_rccl_develop.sh` builds `projects/rccl` for gfx942 inside `rocm/primus:v26.7` in about
2 minutes. For revisions from before mid-August 2026, it adapts two `amdsmi_fabric_info_t` field
accesses to this image's amd_smi header. That code path only reads UALoE fabric info.

## Results

8x AMD Instinct MI308X (gfx942, SPX), 8 processes, P2P on.

| RCCL (loaded `ncclGetVersion`) | Image / build | Result |
|---|---|---|
| 2.30.4 | `rocm/primus:v26.7` (stock) | PASS |
| 2.30.4 | develop `b40d918f58` (2.30.4 sync), built in primus v26.7 | PASS |
| 2.30.7 | develop `4c0dc03b09`, built in primus v26.7 | PASS |
| 2.30.7 | develop `5f38cd0701`, built in primus v26.7 | **FAIL, first bad** |
| 2.31.2 | develop `4bab2283`, built in primus v26.7 | FAIL |
| 2.31.2 | develop `4bab2283` + `experiment_head_gpu_dev_smi_index.diff` | PASS |
| ROCm 10.2 build | `rocm/frameworks-internal:rocm10.2.0a20261007_ubuntu24.04_py3.12_pytorch_2.12.0_9ccf2fd` (latest ROCm build) | **FAIL**|



