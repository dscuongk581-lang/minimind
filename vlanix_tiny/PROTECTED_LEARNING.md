# P0.11M — protected mathematical core
P0.11L CI 45578209491 passed all 3 jobs and 144 smoke tests.
Implicit vs explicit: byte 991 vs 996 /1728, trace 1303 vs
1531 /1728, local 58 vs 60 /64. Explicit full/bounded/reset
32-bit exact 5/70/80 of 128. K and L have run-to-run variability;
deterministic CPU algorithms enabled and environment reported here,
not a guarantee of cross-host bitwise reproducibility.
Decouple local state transitions from readout: reset local context
for each bit using previous predicted carry, then ask the SAME
model to read eight predicted packed states with BOS/ID18 prefix.
The scheduler supplies order and state extraction; no true state
or oracle answer in free inference. Final byte from neural argmax.
Compare ordinary vs guarded training of standalone readout from
gold states plus local rehearsal. Same 120-step foundation, exact
32/32, then same 160 proposed batches of 24 readout and 8 local
examples, 32 targets per batch, optimizer moments, RNG and fixed E.
Guard rejects any proposed step failing full 32-context argmax
check and restores parameters, persistent state, optimizer and
CPU RNG. No retries, no extra training, rejected batch skipped.
Guarded steps may stall; accepted budgets and compute differ.
Local argmax preservation is not distribution conservation or
protection of general skills. Trusted serial CPU callbacks only,
no external transaction or security guarantee. Gold state readout
separate diagnostic; validation traces may duplicate train states
even though operand pairs are disjoint. No test split use.
Cache model predictions for local table, check individual parity,
then use cached table for 8-bit end-to-end score. Wider probes
reuse original state protocol; no learned 16/32-bit final readout.
Report both arms' accepted/rejected steps, min local correct,
neural final byte vs external decode and gold-trace byte scores.
Local RED missing module -> GREEN 4 tests in 0.693s PyTorch
2.10 CPU; full CI pinned torch2.6 Transformers4.57.6 pending.
No growth, GPU or deployment.
