# Patch A — drafter-private cudagraph capture sizes (DSpark single-stream fix)

**Problem.** The DSpark proposer shares the target model's cudagraph capture sizes. With
speculative decoding those sizes are rounded to multiples of `1+k` (k=5 → 6, 12, 18, …), so a
batch-1 draft call dispatches on the 6-bucket, is clipped to the 4 draft rows (`max_num_seqs`),
and the draft MoE processes **20 draft tokens per step for a single stream instead of 5**.

**Fix.** `VLLM_DSPARK_DRAFT_CAPTURE_SIZES=1` installs a drafter-private capture-size view
(`{1,2,4}` up to `max_num_seqs`), captured in `dummy_run`, leaving the target's uniform-decode
FULL graphs untouched. Default OFF = byte-identical to stock behaviour.

**Measured (benchmarks/campaign-2026-08-20):** +3.0% single-stream decode battery, chat +6.4%,
code +6.5%, quality probes byte-identical. Costs ~2.5 GB at profiling for the extra drafter
graphs — run with `--gpu-memory-utilization 0.78` at 909K context.

**Apply.** Bind-mount `v1/spec_decode/dspark_proposer.py` over
`/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark_proposer.py:ro` on **both**
nodes and set the env var on both. Same rank-mismatch warning as every other DSpark switch: the
toggle must be identical across ranks or the collective sequence diverges and NCCL hangs.
