# P0.11A — mathematical learning foundation
Return to learning quality; do not modify infrastructure, growth or guards.
Separate research fixture, opcodes 1 add modulo 256, 2 XOR, 7 AND. These
are curriculum-local meanings; no claim of shared wire protocol with
P0.9C (which did not define opcode 7). All frames are three bytes, BOS+3
IDs in model, one answer ID objective, no EOS. Independent bit carry oracle.
Split by unordered operand pair across ALL three operations, so swapped
or cross-operation pairs cannot leak across splits. Easy = both 0..15;
OOD = both >=128; hard = remainder. Train sees every byte as operand
but never high-high pairs. Easy train96 pairs, hard384; easy validation24,
hard96, OOD96; test reserved, never used in model metrics.
Three arms: random tied embedding mixed order, same random initialization
curriculum (easy two passes then hard two), bit-structured initialization
same curriculum. Each arm sees identical example multiset, same token IDs,
parameter count, batch size32, optimizer AdamW .0005, gradient clip1,
2 epochs, 90 updates. Structured byte rows half norm-squared in 8 bipolar
bit coordinates, remainder original random direction, preserve row norm
and controls; all embedding weights still trainable. This is an explicit
inductive bias, not learned arithmetic, not frozen weights. Same initialization
seed per arm and same order within curriculum arms. Two seeds42/43. No
factorial bit-mixed arm yet; no claim of interaction effects. Report exact
answer accuracy, bit accuracy (invalid IDs count zero), unconstrained 265-way
CE, invalid counts and train-derived majority comparator by family/tier.
OOD baseline uses hard train majority. No test evaluation, no growth,
no production updates or GPU. Next: evaluate effects and compare
frozen vs slow vs trainable tied embeddings only after evidence.
Run python -m vlanix_tiny.math_learning.
