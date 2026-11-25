# P0.11C — novel rule assimilation
P0.11B CI 45564994530: 103 tests passed; frozen E exact, mean
hard AND frozen/slow 48/192 vs full 43/192; OOD AND zero.
Recreate a common old-rule foundation (ADD XOR AND, 90 curriculum
updates + 90 mixed updates with E fixed). OR is opcode 8 in the
curriculum-local grammar; no new token, no answer in prompt.
OR uses the same unordered operand pairs and split assignments as AND,
with independent per-bit oracle. No OR supervision in foundation;
byte 8 itself is not unseen. OR test data remains unused in metrics.
Three E policies crossed with pure OR or 25% rehearsal:
each 32-example update contains the same 24 OR anchor examples,
plus 8 extra OR examples or 8 old training examples sampled without
replacement within each batch. 60 steps per arm; pure sees 1920 OR
targets vs replay 1440 OR + 480 old. Same total compute proxy, not
identical novel-target budget. Same checkpoint weights, optimizer
moments and RNG across arms; all E gradients retained.
Report old/new exact accuracy, bit accuracy, CE, majority and invalid
IDs by tier/family; delta in aggregate accuracy and CE, KL on old
validation, E displacement. No hard gate, no new growth, no GPU.
Old knowledge is weak; reduction in old accuracy is not necessarily
catastrophic forgetting. No claim of compositional zero-shot reasoning.
Two seeds, repeated validation, exploratory. Run
python -m vlanix_tiny.rule_transfer.
