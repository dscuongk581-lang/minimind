# P0.9B — cross-executor conservation
Run cargo build --manifest-path vlanix_binary/Cargo.toml --locked --offline --bin parity_executor
and python vlanix_binary/parity.py vlanix_binary/target/debug/parity_executor
with PyTorch installed (CI uses 2.6 CPU).
Independent Python generator writes VLXEMB01 fixture; Rust restores and
exports it, performs all 265 lookups, byte ID framing for all 256 byte
values, and 8x265 logits for each of two deterministic cases. One uses
small dyadic numbers (exact sums); another uses seeded normal FP32 values.
Require byte-exact snapshot and lookups, exact dyadic scores, tolerances
atol=2e-5/rtol=1e-5 for general FP32, plus shared-matrix mutation parity.
Hashes describe the fixture, not authentication. The executor is an offline
test harness with fixed-size reads, not a general service.
No gradient parity, learning, end-to-end neural core, hardware migration or
optimizer checkpoint verified. No REA dependency. The two executors are
tested on CPU, not different physical architectures. Keep the BPE baseline.
P0.9C: compare machine-native structured binary datasets against byte and
BPE textual controls, at matched token/compute/parameter budgets.
