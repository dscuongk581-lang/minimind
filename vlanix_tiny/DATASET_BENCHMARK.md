# P0.6B — multi-shard screening

P0.6A observed 17 train documents, 1 validation document, 167823/393 target
tokens, zero reserved tokens. Validation was too small for model assessment.

Scan up to 128 rows per shard across four pinned shards, retain up to 500 KB
serialized data per shard, skip rather than truncate documents above 12000
characters. These are pilot selection choices, not representative sampling.
Train one tokenizer on train only, measure identical documents at four
chunk budgets and estimate deterministic next-fit occupancy. No actual
packing speed or training benchmark. Gate requires 10 validation documents,
5000 original validation tokens, 5–20% original token validation fraction,
all shards represented, full vocabulary and zero exact cross-split chunk
overlap. Gate failure is reported, not masked. Passing is only data-screening
readiness, never training permission or GPU readiness. No dataset text or
weights published. Network and cache usage may exceed output cap.

```sh
python -m vlanix_tiny.dataset_benchmark
```
Source attribution/license review and independent held-out evaluation still
required. CPU CI fallback uses same command under resource limits.
