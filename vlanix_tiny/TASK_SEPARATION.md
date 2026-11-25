# P0.11L — explicit readout boundary
P0.11J CI 45575949531 passed all jobs and 136 smoke tests.
Free trace accuracy, 128 paired cases per width:
baseline 120/71/15, local8 105/51/12, both8 91/1/0,
byte8 92/0/0 at 8/16/32 bits. Gold-prefix exact trace
counts are identical because exact-trace success requires no error
at any step; compare per-position errors to diagnose propagation.
Weighted final-byte arms show strong first-error concentration at
bit index 8. Hypothesis: implicit readout at a fixed length competes
with continuing state transitions. Not yet causal evidence.
Compare two new arms at fixed (1,8,8) state/byte/local token weights.
Identical foundation, AdamW moments, RNG, 160 batches, 3840 trace
and 1280 local examples, 27 padded positions and 224 targets per batch.
Implicit final byte at position25, trailing masked pad. Explicit
readout marker ID18 (existing byte9, research grammar only) at
position26; final label moves to that position. No new token or
parameter; both evaluate same unrestricted 265-way argmax.
The extra meaningful token and shifted final-label position are
joint treatment, not isolated semantic-marker effect.
Separate free final-byte and exact trace, gold-prefix readout,
external state decode, local retention and paired width probes
including full/bounded/reset. No new width training, no correction,
no held-out test metrics or autonomous planning claims.
Local Python RED missing module -> GREEN 4/4 via unittest.
Full model CPU CI pending. No GPU, growth or deployment.
