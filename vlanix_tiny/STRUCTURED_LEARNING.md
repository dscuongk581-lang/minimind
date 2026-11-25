# P0.11E — structured binary input
P0.11D CI 45568253544: 112 tests passed, CE 176/1728 vs
CE+bit loss 166/1728 exact validation answers; retain CE.
Compare raw BOS/op/a/b (4 IDs) with BOS/op/a0/b0/.../a7/b7
(18 IDs, LSB first). Both losslessly encode the same query, no carry,
intermediate solution or answer in prompt. IDs 9 and 10 denote byte
values 0 and 1, not new tokens. Same 265-way output and exact byte CE.
Separate research input modes, not a deployed wire format.
Both arms start from identical seeded initial core and bit-structured
E, fixed from initialization; no warmup checkpoint, to avoid raw-input
pretraining bias. Two seeds, same mixed order, 3840 targets/120 updates
per arm, same AdamW and clipping. Report CE, exact and bit accuracy,
invalid predictions and majority by family/tier, prompt token count and
training elapsed time. Longer sequence costs 4.5x prompt tokens and
different FLOPs; no claim of matched compute or a fair speedup.
No held-out test metrics, no growth, no GPU, no changes to runtime.
Exhaustive lossless roundtrip tests on 65536 operand pairs, plus all
opcodes and invalid frames. Run python -m vlanix_tiny.structured_learning.
