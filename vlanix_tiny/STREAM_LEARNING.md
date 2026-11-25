# P0.11H — causal stream layout and local memory
P0.11G CI 45571698606: 124 tests passed. Direct 139/1728 exact
answers, trace 60/1728. Local table retention direct 14/64,
trace 51/64. Fixed E alone does not retain learned behavior.
Both new arms reproduce the 32/32 local foundation with 120 updates.
Then train with identical 24 trace examples and 8 local table examples
per batch, 160 updates, 3840 trace examples and 1280 local examples.
Front: all operand bits before trace. Stream: reveal next original
operand bit pair, then generate one packed state; after eight states
predict final byte. No true carry or answer is supplied at inference;
predicted IDs (including invalid state IDs) are fed back unchanged.
Scheduler still fixes the input ordering and nine output steps.
Both use 26-position training sequences, 224 supervised tokens per
batch, same optimizer/RNG/initial weights/order, frozen tied E.
Local prompts right padded and loss masked after the single target;
causal attention prevents future padding influencing that target.
Loss is token averaged; rehearsal is 25% of examples, only 8/224
of supervised positions. No claim of equal weighting by example.
Both arms have rehearsal, so its causal effect is not isolated.
Compare final byte, trace, joint accuracy, consistency, invalids,
local retention and runtime. No output correction, no autonomous
planning, no held-out test evaluation, no growth or GPU.
Local pure Python RED missing module then GREEN 4 tests passed.
Full CPU CI required. Run python -m vlanix_tiny.stream_learning.
