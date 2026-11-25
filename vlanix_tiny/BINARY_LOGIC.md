# P0.9C — binary mathematical curriculum v1

A separate machine-first research arm. All query/answer payloads are bytes,
not text or executable opcodes. Hex in JSON records is storage only.
Opcode 1: add modulo 256 (a,b). 2: byte XOR (a,b). 3: state update
(a+b)*c modulo 256. 4: next term of modular arithmetic progression
(a,b,c) with equal modular differences. 5: closure of initial fact mask
a under eight adjacency bitmasks (one per bit). 6: Fibonacci-like recurrence
(a,b,n), index n in 2..8, modulo 256. Every answer is one byte.
A reference interpreter and alternative formula/traversal oracle must agree.
No outside corpus. Intended synthetic research fixture, not a trained model.
IDs use SHA-256 of raw query. Split by first operand: 0..127 train,
128..191 validation, 192..255 test. The OOD split is intentional; no exact
query overlap, but related algebraic instances and shortcuts remain possible.
Generator seed 42, 128 rows per family per split (2304 total); deterministic
for the selected Python implementation. Verify all rows for data integrity,
not model selection on test. Do not train on validation or test.
Encode BOS, query bytes mapped +9, answer byte +9, EOS. Query and BOS labels
are -100; shifted CE supervises answer and EOS only. Evaluation should measure
answer exact accuracy, separately from EOS. Inference gets query only.
No symbolic solver at inference. No hidden reasoning trace requirement.
Reference solver is not a security sandbox or arbitrary binary executor.
Limits: tiny fixture, fixed grammar and modular rules, output imbalance
(especially closure), family-specific shortcuts, no proof of reasoning.
Compare matched tasks in natural/formal text and bytes before claiming
sample efficiency. Keep BPE reference. Next: 265-vocab model and
answer-only loss training smoke, fair budget control, retention probe.
Full training/GPU not authorized here.

Run python -m vlanix_tiny.binary_logic
