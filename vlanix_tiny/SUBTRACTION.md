# P0.11W — new rule acquisition, gated on V
V run 45614515604 still in progress at implementation.
Do not assume V passed. On this push the core-verification jobs
run the fixed V recipe again. W needs all four V jobs and smoke
to succeed, then uses the same-run seed-specific artifact, checks
finite_gate, schema, seed, checkpoint SHA and frozen E digest.
SUB raw opcode10 (existing token19), no vocabulary expansion.
Local transition d=a-b-borrow, packed output (d&1)+2*(d<0).
Supervise all eight new contexts; not zero-shot rule discovery.
120 proposed batches, 8 new +32 old contexts +24 scheduled
readout rehearsal frames. Loss .5 new CE +.25 old CE +.25
readout CE +.25 old margin hinge. Exploratory weights.
Adaptive exact guard for 32 old contexts and 64 fixed canonical
readout sentinels from original scheduled training data. This
does not protect all readout frames at every step. Report novel
local learning trace. After training, full 65536 readout argmax
table, old 262144 arithmetic cases, new 65536 ordered SUB cases.
Use cached actual predictions, not oracle arithmetic in rollout.
Require exact all old and new, all 40 local contexts and entire
readout table; failure is research outcome. Save candidate and
report, no deployment. No parameter growth or GPU. Prior
checkpoint remains unchanged; no promotion on failed gate.
Local actual RED import missing, GREEN 4/4 tests in 1.036s;
reference oracle passed all 65536 SUB cases, not model result.
SHA256 subtraction.py f989c7b72db53788827ea4138a455c374f43113b1c2ca17697bc5970a5f3cd8c
SHA256 test_tiny_subtraction.py 09cb55a7b61a2df27a356e6b7a58aaa27b71b8bd4550ed41531e06d0e581b523
Full integration/model tests pending CI.
