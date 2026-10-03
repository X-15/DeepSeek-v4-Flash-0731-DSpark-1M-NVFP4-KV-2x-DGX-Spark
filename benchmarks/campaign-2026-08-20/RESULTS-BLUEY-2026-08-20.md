# RESULTS — Bluey campaign — 2026-08-20 (append-only log)

Baseline: see `BASELINE-BLUEY-2026-08-20.md` (battery mean **63.96**, acceptance 60.23%, 32K prefill ~1,830 tok/s, c4 105.2).
Columns: battery mean (Δ vs baseline) | chat | count | code | prose | tool | accept % | 32K prefill tok/s | 32K TTFT s | c4 aggregate | quality | garble

| experiment | battery mean | chat | count | code | prose | tool | acc% | 32K pf | 32K ttft | c4 | quality | garble |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 63.96 (+0.0%) | 39.04 | 87.12 | 65.74 | 42.10 | 85.81 | 60.23 | 1830 | 25.6 | 105.2 | 6/6 ref | 30/30 |

## Queue (ranked after source recon; one variable per boot; ✓ done · ✗ abort/dropped · — pending)

1. — E1 greedy draft → **noise-floor run** (recon: draft_sample_method is a no-op on this tree — drafter always argmaxes)
2. — E2 k=4 (`num_speculative_tokens:4`) — cuts verify traffic; positions 4–5 rarely accepted on prose
3. — E3 `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1` — one Triton kernel replaces 3 per draft position ×5/step
4. — E4 `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=1` — keeps the fused mHC chain intact at DSpark target layers
5. — E5 `--max-num-batched-tokens 16384` — prefill/TTFT/c4 (long-prefill threshold stays 0)
6. — E6 NCCL env: `NCCL_PROTO=LL128` (small latency-bound collectives) ± agent-C fabric plan (busbw, dual-HCA, system NCCL)
7. — E7 drafter `"enforce_eager":true` — removes the 4× single-stream draft padding, loses draft cudagraph (sign unknown)
8. — E8 `VLLM_USE_B12X_MHC=1 B12X_MHC_MAX_TOKENS=0` — b12x fused mHC kernels (med risk)
9. — E9 `VLLM_USE_B12X_SPARSE_INDEXER=1` (med risk; needle check required)
10. — E10 `VLLM_DSV4_B12X_COMPRESSED_MLA=1` (med-high risk; needle check required)
11. — E11 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024` — prefill at depth
12. — E12 `--prefix-caching-hash-algo xxhash` — TTFT on long prompts (if xxhash present)
13. — E13 k=3
14. — E14 `--max-num-seqs 6` + gmu 0.80 — c4
15. — E15 STACK: all individual winners combined, re-measured as one config (never assumed additive)
   ✗ block-size 128/512 — V4 FlashMLA + indexer kernels require exactly 256
   ✗ cudagraph capture sizes — already rounded to exact multiples of 1+k; nothing to gain
   ✗ KV fp8_ds_mla — label-only on this tree for deepseek_v4 (same 584 B/token, same kernels)
   ✗ confidence scheduler — breaks uniform decode → PIECEWISE graphs + per-step host sync
   ✗ Mia PR #90 — their cap=1→2 never existed on our tree
   SOURCE-LEVEL (code agent, patch files, tested before any mount): (a) separate drafter capture sizes — kills the 4× draft padding; (b) export real draft probs for T>0 requests — production agents run T>0 and currently get one-hot draft probs (acceptance = p_target(tok) instead of exact-match)

## Log

### Fabric plan notes (agent C, pre-experiment)
- The idle `<hca-half-B>` is NOT a second link: GB10's CX-7 exposes each physical QSFP port through two PCIe Gen5 x4 root ports (`0000:01:00.0` / `0002:01:00.0`), each half ~100 Gb/s. Our `NCCL_IB_HCA=<hca-half-A>` uses half the port. NVIDIA's multi-Spark-through-switch playbook addresses both halves; eugr's launcher defaults to both. The "switched fabric leaves it dark on purpose" note was wrong.
- `NCCL_IB_MERGE_NICS=1` alone does not fuse the halves (different PCI domains → no PORT-level merge); fused vNIC needs `NCCL_NET_MERGE_LEVEL=SYS`; unfused both-devices also works (what PR #35 measured).
- Worker is an ASUS GX10: BIOS older than 2026-04 wires `0002:01:00.x` at Gen5 **x2** → check `lspci -vvs 0002:01:00.0 | grep LnkSta` before trusting any dual-HCA number.
- Expected effect: c1 decode ~0 (tiny latency-bound collectives); prefill at depth and c4 are where it shows. Secondary-win candidate only.
- Current env audit: `NCCL_P2P_DISABLE=1` is a no-op with 1 GPU/node; `NCCL_CUMEM_ENABLE=0` no perf impact; `MERGE_NICS=0`/`CROSS_NIC=0` no-ops with one HCA.
- System NCCL (apt 2.30.4 same version, different Build-ID): stability hypothesis, speed-neutral. Low priority tonight.

| E1-greedy (noise floor) | 63.26 (−1.1%) | 39.04 | 87.35 | 66.20 | 39.87 | 83.84 | 59.33 | 1845 | 25.4 | 109.6 | 6/6 MATCH | 30/30 |

- **E1 05:42→05:56 — NO-OP BY DESIGN → noise floor.** Recon proved `draft_sample_method` is ignored by the DSpark drafter (always argmax), so this is a second boot of the baseline config. **Run-to-run spread: battery ±1%, chat/count/code <1%, prose ±5%, tool ±2%, c4 ±4%, 32K prefill ±1%.** A result must clear these to count. Also confirms the fresh-`docker run` boot path is stable (350 s both times).

| E2-k4 | ABORT | — | — | — | — | — | — | — | — | — | — | — |

- **E2 06:11→06:18 — ABORT: KV cache check.** Head 8.87 GiB / worker 6.25 GiB available vs 6.76 GiB needed for one 909K request. The message's own arithmetic (6.25 GiB ≈ 773K tokens) shows the baseline pool (1.32M tokens) is only ~10 GiB — **the baseline sits ~3 GB from the KV cliff at gmu 0.76**. k=4 re-rounds capture sizes to multiples of 5 (max 50 vs 48) and costs just enough to tip it. Consequence: any experiment that adds activation/graph memory (k changes, larger batched-tokens) needs gmu headroom first. Not a speed result; not counted.
- **Runner defects found and fixed before E2 (both silently neutralized E1's config — E1 is therefore a pure baseline rerun):** (1) `fs.protected_regular` denies root an `O_CREAT` open on another user's file in sticky `/var/tmp` → serve-script write failed on the head; now written via root temp + `cp`. (2) the sed expression lost its quotes across the SSH hop → worker substitution no-op; now `printf %q`-quoted. (3) env overrides were inserted *before* the baseline env → Docker kept the baseline value; now inserted after. Runner prints each node's effective k/batch before every boot.
- **Fabric prep done (no boot):** both CX-7 halves on both nodes link at 200G, both PCIe Gen5 **x4** (no GX10 x2 trap), MTU 9000, LLDP shows both halves on the same the switch port (one cable). Second halves addressed via NetworkManager (`<nm-profile-second-half>` → <second-half-ip>/.4; manual `ip addr` was reverted by NM within seconds). GID index 3 = RoCE v2 + correct IPv4 on all four HCAs. Jumbo ping passes. Dual-HCA is ready as env-only: `NCCL_IB_HCA=<hca-half-A>,<hca-half-B> NCCL_IB_MERGE_NICS=1 NCCL_CROSS_NIC=1 NCCL_IB_ADDR_RANGE=<fabric-cidr>/23`.

| E0b-gmu80 (control) | 63.86 (−0.2%) | 40.00 | 86.76 | 68.25 | 39.68 | 84.59 | 59.75 | 1740 | 27.0 | 107.3 | 6/6 MATCH | 30/30 |

- **E0b 06:32→06:53 — gmu 0.76→0.80: decode speed-neutral (within noise), KV pool 1.32M→1.79M tokens (+35%), quality intact. FLAGGED:** 3 driver `NV_ERR_NO_MEMORY` lines at 06:37:02, inside the memory-profiling dummy forward (the allocator retried; boot and serve were clean, 0 watchdog triggers). 32K prefill −5% and TTFT +5% — outside prefill's ±1% noise, plausibly page-cache pressure with 10–13 GB free while serving. **Not adopted as a blanket floor; 0.78 is the headroom step for memory-hungry experiments, 0.80 stays a flagged option.**

| E2b-k4-gmu78 | 60.06 (−6.1%) | 39.90 | 77.08 | 63.89 | 42.25 | 77.19 | 66.22 | 1853 | 25.2 | 101.3 | **5/6 DIFFER** | 30/30 |

- **E2b 06:55→07:15 — k=4 LOSES. Battery −6.1%; count −11.5%, tool −10%, code −3%, chat/prose flat.** Acceptance rose 60→66% (fewer positions to reject) but accepted tokens/step fell further than the per-step cost saved — on easy content k=5's 5th position earns its keep. One stable probe changed → "different model" on top of the loss. **k=3 dropped from the queue** (strictly worse by the same mechanism). Driver OOM lines again during profiling at 0.78 (5 lines), boot clean; KV pool 1.48M.

| E3-fused-markov | 63.46 (−0.8%) | 38.21 | 85.99 | 67.87 | 41.43 | 83.78 | 59.73 | 1855 | 25.2 | **114.4 (+8.8%)** | 6/6 MATCH | 30/30 |

- **E3 06:56→07:16 — `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1`: single-stream neutral (within ±1%), quality intact, c4 aggregate +8.8% — double the c4 noise band.** Mechanism fits: one Triton block-argmax replaces three kernels per draft position ×5 per step; launch overhead is proportionally larger at batch 4. Tentative c4 win → carried into the final STACK run for confirmation. KV pool reported 1.14M (vs 1.32M baseline at the same gmu) — the fused path's JIT/buffers cost ~1.5 GB at profiling; note for headroom.

| E4-defer-capture | 62.71 (−2.0%) | 36.22 | 86.41 | 66.13 | 40.38 | 84.42 | 58.15 | 1851 | 25.3 | 105.2 | 6/6 MATCH | 30/30 |

- **E4 07:17→07:37 — `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=1`: no gain** (−2.0% battery, chat −7% the only class outside noise), c4 flat, quality intact. Not carried forward.

| **E5-patchA** | **65.89 (+3.0%)** | 41.53 | 89.64 | 70.03 | 41.00 | 87.25 | 60.15 | 1854 | 25.2 | 102.8 | 6/6* | 30/30 |

- **E5 07:32→07:46 — Patch A (drafter capture sizes {1,2,4}, env `VLLM_DSPARK_DRAFT_CAPTURE_SIZES=1`, mount `v1/spec_decode/dspark_proposer.py`): FIRST WIN. Battery +3.0% (clear of the ±1% floor); chat +6.4%, code +6.5%, count +2.9%, tool +1.7%, prose −2.6% (inside its ±5%). Prefill and c4 flat.** Mechanism as recon predicted: a batch-1 draft no longer runs padded to the 6-bucket / 4 rows (20 draft tokens per step) — it runs 5.
- *Quality: the compare run flipped `seq` once (`…128` → `…128, 256, 5…`, a stop-position difference, both correct). Re-tested on the same lane: `seq` is **unstable** under Patch A (2 back-to-back runs differ) and 4 further greedy runs all returned the baseline text; all other stable probes byte-identical. Classified quality-neutral — the lane already carries 2/8 non-deterministic probes at t=0 (trap-60 class), and this is that, not a systematic change. Evidence: `campaign/ref-patchA.json`, `E5-patchA.compare.txt`.*
- KV pool 1.01M at gmu 0.76 (vs 1.32M baseline): the extra drafter graphs cost ~2.5 GB at profiling. Worth knowing for the STACK run's headroom (use gmu 0.78 there).

| E6-dualHCA | 62.77 (−1.9%) | 40.44 | 86.41 | 65.51 | 37.16 | 84.31 | 58.25 | 1861 | 25.1 | 103.1 | 6/6* | 30/30 |

- **E6 07:48→08:08 — dual-HCA (`NCCL_IB_HCA=<hca-half-A>,<hca-half-B>`, MERGE_NICS=1, CROSS_NIC=1, /23 range): NULL.** NCCL confirmed both devices (`NET/IB : Using [0]<hca-half-A> [1]<hca-half-B>`, channels alternating NET/IB/0 and /1), yet decode, 32K/128K prefill and c4 all within noise. **The interconnect is not the bottleneck for this workload** — per-step collectives are latency-bound and prefill is compute-bound on GB10. Not carried. (Second-half addressing left in place via NetworkManager; harmless, reversible with `nmcli con mod <nm-profile-second-half> ipv4.method link-local`.) *`seq` probe flipped the same way as E5 — the unstable probe, same classification.*

| E9-b12x-mhc | ABORT | — | — | — | — | — | — | — | — | — | — | — |

- **E9 08:10→08:15 — `VLLM_USE_B12X_MHC=1`: ABORT at init** — `RuntimeError: Can't export tensors that require gradient, use tensor.detach()` (the b12x mHC integration passes an autograd-tracked tensor to a DLPack export). Broken path in this image, not a performance verdict. Reported for the fork maintainers.

| E10-b12x-indexer | 62.64 (−2.1%) | 37.98 | 86.06 | 65.53 | 39.34 | 84.29 | 57.82 | 1781 | 26.2 | 106.3 | 6/6 MATCH | 30/30 |

- **E10 08:17→08:37 — `VLLM_USE_B12X_SPARSE_INDEXER=1`: no gain** (decode −2.1%, 32K prefill −2.7%, c4 flat), 4 driver-OOM lines at profiling. Not carried.
- **T=0.7 reference (measured on the E10 lane, 10 prose prompts, top_p 0.95): acceptance 27.62% (1779/6440), 33.9 tok/s** — versus ~40% / 42 tok/s for prose at t=0. This is the sampled-request penalty from one-hot draft probabilities (recon finding #2). Production agents run at T>0; this is the number Patch B targets.

| E11-patchB | 62.33 (−2.5%) | 38.76 | 86.11 | 65.23 | 39.34 | 82.22 | 57.91 | 1858 | 25.2 | 103.7 | 6/6* | 30/30 |

- **E11 08:40→09:02 — Patch B (real draft probabilities for sampled requests): mechanism works, payoff too small.** t=0 battery within noise as designed (the greedy graph is untouched; `seq` flip again = the unstable probe). **T=0.7 prose: acceptance 27.62% → 29.96% (+2.3 pts, +8% relative), throughput 33.9 → 34.0 tok/s — flat.** The full-vocab gathers per position (~1.3 MB/step at B=1 over RoCE) cost roughly what the acceptance gain earns. The drafter's distribution is flat relative to the target, so Σmin(p,q) barely exceeds p(argmax q). Not carried. Follow-up if wanted: `VLLM_DSPARK_DRAFT_TEMPERATURE_SCALE` sweep (0.5/0.7) to sharpen q — one boot each.

| E7-batched16k-gmu78 | ABORT | — | — | — | — | — | — | — | — | — | — | — |

- **E7 09:05→09:11 — `--max-num-batched-tokens 16384` @ gmu 0.78: ABORT, KV check** (per-request requirement rose to 10.02 GiB with the larger chunk buffers). Not retried: prefill has sat within ±2% across every experiment tonight and the interconnect is ruled out — it is compute-bound on GB10, and a bigger chunk is unlikely to change that.

| **E15-STACK (WINNER)** | **65.10 (+1.8%)** | 39.38 | 89.77 | 66.59 | 42.72 | 87.06 | 60.25 | 1818 | 25.7 | **110.4 (+5.0%)** | **6/6 MATCH** | 30/30 |

- **E15 09:13→09:35 — STACK = Patch A + `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1` @ gmu 0.78: the final config.** Battery +1.8% (Patch A alone measured +3.0%; the two are within the ±1% floor of each other), c4 +5.0%, 32K prefill −0.7% (noise), KV pool 1.38M (+5% vs baseline) with the 0.78 headroom, **every stable probe byte-identical including `seq`**, garble 30/30. Two driver-OOM lines at profiling (the known 0.78 pattern; boot and serve clean, 0 watchdog triggers). **Bluey is left serving this.**

---

## FINAL SUMMARY

### Scoreboard (primary metric = single-stream decode battery mean, t=0; noise floor ±1%)

| config | battery | Δ | c4 | Δ | quality | verdict |
|---|---|---|---|---|---|---|
| baseline (gmu 0.76, k=5, probabilistic) | 63.96 | — | 105.2 | — | 6/6 | reference |
| E1 noise-floor rerun | 63.26 | −1.1% | 109.6 | +4.2% | 6/6 | defines the floor |
| E0b gmu 0.80 | 63.86 | −0.2% | 107.3 | +2.0% | 6/6 | neutral, +35% KV, flagged at the memory edge |
| E2 k=4 @0.76 | ABORT | | | | | KV cliff |
| E2b k=4 @0.78 | 60.06 | **−6.1%** | 101.3 | −3.7% | 5/6 | LOSS — k=5 is right; k=3 dropped |
| E3 fused Markov argmax | 63.46 | −0.8% | 114.4 | **+8.8%** | 6/6 | c4 candidate → stacked |
| E4 defer target capture | 62.71 | −2.0% | 105.2 | 0% | 6/6 | no gain |
| **E5 Patch A (drafter sizes)** | **65.89** | **+3.0%** | 102.8 | −2.3% | 6/6* | **WIN → stacked** |
| E6 dual-HCA | 62.77 | −1.9% | 103.1 | −2.0% | 6/6* | NULL — interconnect not the bottleneck |
| E9 B12X mHC | ABORT | | | | | broken path in image |
| E10 B12X sparse indexer | 62.64 | −2.1% | 106.3 | +1.0% | 6/6 | no gain |
| E11 Patch B (draft probs, T>0) | 62.33 | −2.5% | 103.7 | −1.4% | 6/6* | T=0.7 accept 27.6→30.0%, tok/s flat — not carried |
| E7 batched 16K @0.78 | ABORT | | | | | KV cliff; prefill is compute-bound anyway |
| **E15 STACK = A + fused-Markov @0.78** | **65.10** | **+1.8%** | **110.4** | **+5.0%** | **6/6** | **WINNER — left serving** |

*\* one `seq` probe flip, shown by re-test to be lane instability (see E5 notes), not a systematic change.*

### What moved the number, and what didn't

- **The one real single-stream lever was structural, not a flag:** the DSpark drafter shares the target's cudagraph capture sizes, which are rounded to multiples of (1+k); a single stream's draft was running padded to the 6-bucket and clipped to 4 rows — **20 draft tokens per step instead of 5**. Patch A (drafter-private capture sizes {1,2,4}) recovers +3% single-stream, most on chat/code. This is a fork bug, not a tuning knob, and it generalizes to every DSpark deployment on this tree.
- **Fused Markov argmax** (one Triton kernel per draft position instead of three) is neutral single-stream and worth +5–9% at c4, where launch overhead stacks.
- **Nothing else moved decode outside noise**: k≠5 loses, B12X alternative kernels are broken or neutral, deferred capture neutral, dual-HCA null, real draft probs neutral on throughput.
- **Prefill never moved** (±2% across all 10 measured configs, 1.7–1.9K tok/s at 32K–128K). It is compute-bound on GB10; the interconnect was explicitly ruled out (both CX-7 halves in use, no change).
- **The memory margin is the hidden constraint**: at gmu 0.76 / 909K the baseline sits ~3 GB from the KV cliff; k=4, 16K batched tokens and Patch A's extra graphs all needed 0.78. 0.80 works but profiles at the physical edge (driver allocation retries during the profiling pass).

### Reproduction recipe for the winner

```
# serve line: identical to baseline except
--gpu-memory-utilization 0.78

# env, set identically on BOTH nodes
VLLM_DSPARK_DRAFT_CAPTURE_SIZES=1
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1

# bind mount on BOTH nodes (read-only), alongside Patch 3 + Patch 4
patches/A-drafter-sizes/v1/spec_decode/dspark_proposer.py
  -> /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark_proposer.py:ro

# boot: fresh `docker run` on both nodes (never docker restart/start on an old container),
# drop caches first, worker first, head ~25 s later
```

Patch A is a byte-complete replacement of the fork's `dspark_proposer.py` with the new behavior behind `VLLM_DSPARK_DRAFT_CAPTURE_SIZES` (default off = identical to stock). CPU-tested for batch 1/2/4/6 dispatch and slot bounds; live-validated above.

### Measurement notes (so nobody re-learns them)

- Every number: non-streaming, `usage.completion_tokens`, unique nonce at the front of every prompt, five prompt classes never pooled, decode via prefill-floor subtraction, t=0. Noise floor from a literal rerun of baseline.
- `draft_sample_method` is a no-op for DSpark on this tree (drafter always argmaxes) — don't A/B it.
- At T>0, acceptance is materially lower than at t=0 (prose 27.6% vs ~40%) because the drafter exports one-hot probabilities; Patch B fixes the mechanism but the gathers cost what they earn at k=5 — a temperature-scale sweep is the open follow-up.
- Two fork issues to report upstream: `VLLM_USE_B12X_MHC=1` crashes at init (`Can't export tensors that require gradient`); and the drafter capture-size sharing (Patch A) is a genuine performance bug.

| E12-compressed-mla | ABORT (wrong output + crash) | — | — | — | — | — | — | — | — | — | **0/6** | — |

- **E12 09:12→09:19 — `VLLM_DSV4_B12X_COMPRESSED_MLA=1` @0.78: BROKEN.** Served, then **0/6 stable probes matched** (every answer different), then a `CUDA error: device-side assert triggered` during the garble gate killed the engine. The B12X alternative-kernel family is now fully accounted for on this image: mHC crashes at init, sparse indexer neutral, compressed MLA produces wrong output and crashes. None are usable.

| E15b-STACK-rerun | 64.80 (+1.3%) | 40.71 | 89.70 | 67.18 | 41.36 | 85.03 | 59.29 | 1637† | 28.6† | **120.0 (+14%)** | 6/6 MATCH | 30/30 |

- **E15b 09:22→09:43 — STACK re-measured (also the restore after E12): decode +1.3% (vs +1.8% first run — consistent), quality 6/6 again, c4 +14%.** †32K prefill read −10.6% inside the snapshot (reps 1709/1565) while 128K was normal (1625/1627); **three fresh 32K probes on the same lane immediately after: 1872 / 1774 / 1822 — baseline level.** Transient, not the config. Prefill verdict for the STACK: neutral. KV pool 1.47M.

| E13-STACK+LL128 | 64.89 (+1.5%) | 39.96 | 87.53 | 70.16 | 40.64 | 86.14 | 60.00 | 1752 | 26.7 | 111.6 | 6/6* | 30/30 |

- **E13 09:45→10:06 — STACK + `NCCL_PROTO=LL128 NCCL_ALGO=Ring`: no additional gain** (identical to STACK within noise; prefill −4%; 10 driver-OOM retries at profiling, the most of any boot). Interconnect axis closed: bandwidth (dual-HCA) null, protocol (LL128) null. *`seq` flip = the unstable probe.*

| E14-patchB-scale0.5 | 63.05 (−1.4%) | 38.22 | 86.48 | 68.75 | 39.86 | 81.95 | 59.50 | 1820 | 25.7 | 102.8 | 6/6 MATCH | 30/30 |

- **E14 10:08→10:30 — Patch B with `VLLM_DSPARK_DRAFT_TEMPERATURE_SCALE=0.5`: T=0.7 prose acceptance 29.17% (scale 1.0: 29.96%; baseline one-hot: 27.62%), 33.2 tok/s — no throughput gain at either scale.** Patch B closed: the mechanism is correct and worth ~+2 acceptance points for sampled requests, but at k=5 the full-vocab gathers cost what they earn. t=0 battery untouched, 6/6.

| E16-STACK+indexer1024 | 63.16 (−1.3%) | 37.46 | 88.80 | 66.33 | 40.57 | 82.62 | 57.38 | 1807 | 25.9 | 106.2 | 6/6* | 30/30 |

- **E16 10:32→10:53 — STACK + `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024`: no gain.** 128K prefill 1622.8 tok/s (baseline 1,622) — the indexer logits budget is not the limiter at this depth. Not carried. **Queue exhausted.**

## Deployment (Phase 2)

- **Bluey pair**: winner restored 10:29–10:35 UTC, verified 6/6 probes, 30/30 garble, count-to-300 ≈ 84 tok/s. Launchers: `/var/tmp/rebuild-{head,worker}.STACK-WINNER.sh`, serve script `/var/tmp/ds4cmd.sh.STACK-WINNER`.
- **Reddie pair (production, 512K)**: winner deployed 10:37–10:44 UTC via `/var/tmp/run-winner.sh <0|1>` on both nodes, gmu 0.76→0.78. **7/7 stable probes byte-identical to Reddie's own reference; count-to-300 87.8 / 87.1 / 89.3 tok/s vs 78–81 before (+10%)**; KV pool 1.35M tokens.
- Operator runbook: `the operator runbook` (project folder).
