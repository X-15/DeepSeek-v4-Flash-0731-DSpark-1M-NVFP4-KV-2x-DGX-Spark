# BASELINE — Bluey pair, DeepSeek-V4-Flash-0731 DSpark — 2026-08-20

**This file is the fixed reference for the optimization campaign. It is never rewritten.**
Every experiment is compared against these numbers, measured on this exact config,
with this exact harness. Raw JSON: `snap-baseline.json`, quality ref: `ref-bluey-baseline.json`.

## Config fingerprint (what produced these numbers)

```
model        /models/ds4-0731-abliterated  (FP8 e4m3 weights, 72,317 tensors, 48 shards)
serve        --tensor-parallel-size 2 --kv-cache-dtype nvfp4_ds_mla --block-size 256
             --max-model-len 909312 --max-num-seqs 4 --max-num-batched-tokens 8192
             --gpu-memory-utilization 0.76 --enable-prefix-caching --enable-flashinfer-autotune
             --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
             --tokenizer-mode deepseek_v4 --distributed-executor-backend mp
             --default-chat-template-kwargs '{"thinking":false}'   (async scheduling on, chunked prefill on)
image        vllm-dspark-runtime:<stage-c tag>  id sha256:c2560d5a...   vLLM 0.21.1rc1.dev339+g1967a5627bc3
patches      Patch 3 (scheduler.py) + Patch 4 (dspark.py) bind-mounted on BOTH nodes
nccl         pip-bundled 2.30.4+cuda13.2     driver 580.142   CUDA 13.0
fabric       TP=2 over RoCE v2, one HCA per node (<hca-half-A>), GID 3, NCCL_IB_MERGE_NICS=0,
             NCCL_P2P_DISABLE=1, NCCL_CUMEM_ENABLE=0, NCCL_CROSS_NIC=0.  A second HCA (<hca-half-B>) is link-UP and unused.
fork env     VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_WO_PROJECTION=1 VLLM_TRITON_MLA_SPARSE=1 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
             VLLM_DSPARK_LOCAL_ARGMAX=1 VLLM_DSPARK_REPLICATE_MARKOV_W1=1 VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
             VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0 VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
             VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0 VLLM_DSV4_B12X_COMPRESSED_MLA=0 VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0
KV pool      1,320,949 tokens  → 1.45× concurrency at 909,312/request
boot         fresh `docker run` both nodes (worker first), caches dropped: serving in 350 s
watchdog     dgx-anti-oom threshold 3 GB, 0 triggers during the snapshot
```

## Methodology (the traps this harness is built against)

- **Non-streaming only**, `usage.completion_tokens` over wall clock. Counting SSE chunks under spec decode under-reports ~4×.
- **Unique nonce at the FRONT of every prompt.** Repeated prompts hit the prefix cache and inflate by up to 9×.
- **Five prompt classes reported separately, never pooled silently.** Acceptance is content-dependent (count ≈ 97%, prose ≈ 40%).
- **Decode tok/s = (completion−1) / (total − prefill_floor)**, the floor measured per prompt with `max_tokens=1`.
- Temperature 0 throughout (deterministic target; A/B-comparable across experiments).
- 5 reps per class; prefill 2 reps per depth; concurrency jobs = [count, prose, count, prose][:c].
- Quality: 8 greedy probes run twice; only byte-identical ones count (6/8 here). An experiment that changes any stable probe is **"different model"**, recorded, never a win.

## THE NUMBERS TO BEAT

### Single-stream decode (primary metric)

| class | rep1 | rep2 | rep3 | rep4 | rep5 | **mean** | median | peak |
|---|---|---|---|---|---|---|---|---|
| chat  | 38.37 | 37.68 | 40.15 | 39.75 | 39.25 | **39.04** | 39.25 | 40.15 |
| count | 88.15 | 86.88 | 86.80 | 86.61 | 87.15 | **87.12** | 86.88 | 88.15 |
| code  | 66.87 | 67.94 | 62.05 | 66.61 | 65.25 | **65.74** | 66.61 | 67.94 |
| prose | 44.10 | 41.21 | 41.62 | 44.14 | 39.42 | **42.10** | 41.62 | 44.14 |
| tool  | 91.29 | 83.39 | 84.04 | 83.32 | 87.03 | **85.81** | 84.04 | 91.29 |

**BATTERY MEAN: 63.96 tok/s**  ·  **draft acceptance over the battery: 60.23%**

### Prefill / TTFT (must not regress)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 1K   | 1,476 / 1,489     | 1.66 / 1.50     | 892 / 990     |
| 8K   | 11,713 / 11,685   | 9.66 / 6.16     | 1,212 / 1,898 |
| 32K  | 46,894 / 46,878   | 26.13 / 25.12   | 1,794 / 1,866 |
| 128K | 186,959 / 187,180 | 115.74 / 114.90 | 1,615 / 1,629 |

### Concurrency (must not regress)

| streams | total tokens | wall (s) | aggregate tok/s | per-stream |
|---|---|---|---|---|
| c1 | 400   | 4.8  | **82.7**  | 82.7 |
| c2 | 697   | 10.4 | **67.1**  | 33.5 |
| c4 | 1,464 | 13.9 | **105.2** | 26.3 |

(c2/c4 mix count+prose, so per-stream is lower than c1's count-only — compare like with like across experiments.)

### Quality reference

6/8 probes byte-stable at t=0: count, fact, logic, list, json, seq. (math and prose were non-deterministic on this lane at t=0 and are excluded from the verdict.) Garble gate: 30/30 clean.

## Win criteria

- **WIN**: battery mean > 63.96 with all 6 stable probes identical, garble 30/30, and no regression > 5% on 32K prefill tok/s or c4 aggregate.
- **SECONDARY WIN**: ≥10% on 32K prefill or c4 aggregate with battery mean within 2% and quality intact.
- **DIFFERENT MODEL**: any stable probe changes → recorded, excluded from wins regardless of speed.
- **ABORT**: engine failure, no-serve in 25 min, driver `NV_ERR_NO_MEMORY`, or watchdog trigger → revert to baseline.
