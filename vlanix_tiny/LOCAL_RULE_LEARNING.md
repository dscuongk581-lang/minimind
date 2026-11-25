# P0.11F — local transition learning
Phase 1 parameter budget frozen at 5,970,176; no organic growth.
P0.11E CI 45569558656 passed 116 tests; raw 163/1728 and bit
planes 187/1728 exact validation answers, neither solves OOD ADD.
Train the same 6M model from bit-structured initialization with fixed E.
A local prompt is BOS/op/a_bit/b_bit/carry_in, target byte packs
result_bit + 2*carry_out. Four ops times eight local contexts = 32
fully supervised contexts, repeated as a full batch for 120 updates.
The oracle is used only for training labels and scoring.
An external scheduler walks eight positions LSB-first, feeding its
own predicted carry into the next context, assembling low 8 result bits.
No arithmetic rule or true carry is consulted by the scheduler. The
scheduler does encode the recurrence/bit decomposition and discards
overflow modulo 256. Model inference is unrestricted 265-way argmax;
IDs outside byte values 0..3 abort. No separate head or new weights.
Compute one model-predicted table and verify parity against individual
model calls on all 32 contexts; use the table to exhaustively compose
all 262144 ordered operand/operator combinations. The exhaustive
score includes every pair, not a held-out split. Local contexts were
all trained; success shows learned finite transitions and external
composition, not generalization to unseen local rules, or autonomous
byte reasoning. Local test of scheduler using true transition table
is only a software reference. Compare prior P0.11E as historical
context, not controlled equal-compute experiment: same updates and
targets, different label information, input lengths, distinct samples,
and 8 inference transitions vs 1 byte call (cache amortizes calls).
Report actual local accuracy, exhaustive accuracy, training time,
embedding equality and serial/cache parity. No GPU, growth or deployment.
