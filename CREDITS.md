# Credits

This repo combines several public efforts. Please credit the upstream authors
when reusing the recipe, the patch, or benchmark numbers.

## DSpark Concurrency Patch

The in-server DSpark concurrency breakthrough comes from Keys / drowzeys:

- Repo: https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- Tested commit in this repo: `7e4d94bbcec95223550517c0fa9244e59f9f6483`

Keys' patch fixes the two core blockers for `max_num_seqs > 1`:

- Request-stable DSpark main-KV slots, so persistent DSpark draft KV follows
  request identity instead of condensed vLLM batch-row position.
- Ragged `query_start_loc` handling for real independent-arrival batches where
  prefill and decode rows mix in the same scheduler step.

The validated concurrency numbers in this repo depend directly on that patch.

## DSpark Cold-Start Garble Root-Cause Fix (Patch 3)

The scheduler-level root cause of the cold-resume garble (prompt echo / leaked
tool-schema text at the start of a reply on long resumed conversations) was tracked
down and fixed as a collaboration between **Roady001** and **Fable**:

- **Roady001** — reported the issue (that the 2026-07-03 launch/config change only
  reduced the symptom and did not address the root cause) and independently validated
  the final fix on his own 2x DGX Spark, confirming the garble is gone without any of
  the earlier config workarounds.
  Issue: https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/issues/3
- **Fable** — the root-cause analysis and the patch: a guard in
  `Scheduler.update_from_output` so spec-token placeholders are only resized on genuine
  decode steps (`new_token_ids` non-empty, `not request.is_prefill_chunk`,
  `status == RUNNING`) — never on a mid chunked-prefill final chunk or a preempted request.

This is the actual root cause of the cold-resume prompt-echo / tool-schema garble, not
the launch/config changes, which only reduced the symptom.
Fix commit: https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/commit/e83606a
See `docs/PATCHES.md` (Patch 3) for the full analysis.

## CUDA-Graph Capture-Size Fix (concurrency throughput)

**Wpnx330** found and fixed a silent throughput cliff: `--max-cudagraph-capture-size`
must be a multiple of `(num_speculative_tokens + 1)`, so passing a raw `MAX_NUM_SEQS`
(6) with spec=3 floored the captured size to 4 — enough for one active request. Any
concurrency above that fell off the captured CUDA-graph path into eager/piecewise and
throughput collapsed to <1 tok/s. Fix: `--max-cudagraph-capture-size $((MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)))`.

- PR: https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/5

## DSpark vLLM Integration

Rafael Caricio published the DSpark vLLM integration and deployment work this
recipe builds on:

- https://github.com/rafaelcaricio/vllm/pull/1
- https://github.com/rafaelcaricio/spark_vllm_docker/pull/1

## Model And Runtime Work

Fraser Price published the DeepSeek V4 Flash DSpark model/runtime work used by
this recipe:

- https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark
- https://github.com/fraserprice/dspark-vllm

## Two-Node DGX Spark Packaging

MiaAI-Lab published the two-node DGX Spark packaging and launch lineage this
repo builds from:

- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

## Upstream Foundations

This work also relies on:

- vLLM
- FlashInfer
- NVIDIA CUDA/NCCL/Blackwell tooling
- DeepSeek V4 Flash
- DeepSeek-AI DeepSpec / DSpark speculative decoding research

## TonyD2Wild Contribution

This repo contributes the validated 2x DGX Spark NVFP4-KV recipe, Stage A/B/C
runtime packaging, sanitized two-node launch flow, application of Keys'
concurrency patch to the NVFP4 profile, and benchmark artifacts from the
validated runs.

## License Notes

Repo-local scripts and docs are MIT licensed via `LICENSE`.

The vLLM overlay files and `patches/keys-concurrency.patch` are vLLM/DSpark
derived and retain their Apache-2.0 lineage from the upstream sources and
Keys' patch repo. Model weights, base images, CUDA/NCCL, FlashInfer, TileLang,
and Triton are separate upstream artifacts with their own licenses and terms.

## 0rand

- Parameterized the API port (`VLLM_PORT`, PR #1).
- Independently identified MTP=5 speculation garbling and proposed the MTP=3 default in PR #1 (2026-06-30) — four days before the 2026-07-03 garble fix adopted the same value on main. Early, correct call.

## paulbrav

- Reported the long-context engine-death crash (#2) with a clean deterministic repro AND shipped the fix (PR #4: sparse-indexer gather guard + stale draft-KV slot clamp).
- Producer-side instrumentation of the slot-corruption mechanism (canary readback, 12h compute-sanitizer, graph-replay-private-state localization) and a debug-tooling dead-ends writeup (cuda-gdb/coredump limits on GB10/sbsa) that materially informs #6.

## DaveCharland

- Reported and characterized the episodic soft-failure (#6): real output tokens parsing to empty/thinking-only content under sustained agent load, with a deterministic within-episode repro and full spec-acceptance-collapse telemetry.
