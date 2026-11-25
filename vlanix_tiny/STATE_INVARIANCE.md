# P0.11K — bounded state diagnostics
P0.11I CI 45575315380 both jobs passed. Across two seeds:
baseline 87/1728 final and 1546/1728 exact trace;
byte8 941/1728 final and 1293/1728 trace;
local8 59/1728 final and 1528/1728 trace;
both8 949/1728 final and 1208/1728 trace.
Local retention 44/64,42/64,61/64,61/64 respectively.
Tradeoff remains; no winning policy selected.
P0.11J smoke 136 tests passed; width job pending at implementation.
K extends same paired frames and same trained model with two inference
interventions. Bounded8 keeps BOS/op, latest seven complete
bit-pair/predicted-state triplets and current pair (max 25 IDs).
No gold state or correction. At width8 require parity with full rollout.
Reset uses the original P0.11F local format BOS/op/a/b/carry (5 IDs),
where carry is extracted from the previous predicted packed state.
Invalid packed state aborts that row, with remaining states marked None.
No external arithmetic oracle, but external schedule and state decode
are part of the interface. Full, bounded and reset compare different
context semantics, not an isolated RoPE intervention; resets remove
history and change format. No new optimizer updates or parameters.
Report all three width accuracy/error diagnostics by op on the same
64 paired frames per width per seed and arm. No final-byte readout
claim for wider widths; decode is external. No held-out test metrics.
Local actual RED missing module -> GREEN 4 tests via unittest.
Full CPU CI pending. No GPU, growth or deployment.
