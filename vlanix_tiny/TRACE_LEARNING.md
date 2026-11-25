# P0.11G — learned mathematical composition
P0.11F CI 45570696425: 120 tests pass, both seeds 32/32 learned
transitions and 262144/262144 correct compositions per seed.
The external scheduler encodes the recurrence and bit assembly.
P0.11G holds the model at 5,970,176 parameters, vocabulary 265,
hidden width 256, tied E fixed from initialization. Reproduce P0.11F
local foundation (120 full-table updates), require 32/32 before branching.
Both arms get same weights, AdamW state, RNG and 3840 examples in
120 updates, same 18-token bit-plane prompt. Direct predicts one
byte. Trace predicts 8 packed bit/carry tokens then a final byte.
Teacher forcing is used only in training: shifted causal CE on the
target suffix, prompt positions excluded. Trace labels use oracle
in training. Evaluation is greedy unconstrained 265-way generation,
no true carry, no oracle, no output mask, no correction or byte
assembly in rollout. It simply appends model-predicted tokens, then
separately scores final byte, exact trace, internal consistency and
joint correctness, with invalid trace IDs counted. Fixed trace length
and order are provided, not learned stopping or planning. Report
local table retention, validation by rule/tier, training/inference
time, token counts and unchanged E. Validation operand pairs are
disjoint from training pairs; local transition contexts are not.
Trace has nine times the target count and longer input, so this is
matched examples/updates, not matched compute or supervision.
No test metrics, growth, GPU or runtime changes.
Local pure Python RED (missing module), GREEN 4 tests passed.
Full model CPU CI is required before any performance claim.
Run python -m vlanix_tiny.trace_learning.
