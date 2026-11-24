# P0.7A — English procedural logic fixtures

Four original programmatic task families, 1–256 examples per family per split.
Train, validation and test are generated separately; instance identity hashes
family and canonical arguments. Arithmetic holds out operand ranges, Boolean
holds out expression operators, state transitions hold out sequence lengths,
rule closure holds out chain depths. Missing evidence is unknown, not false.
No external datasets copied. Answer solver and canonical row reconstruction
validate integrity; not an independent formal proof verifier. Templates and
distractors repeat, so these are small engineering fixtures, not evidence of
general reasoning. The Boolean and rule distributions need further balancing.
No full reasoning traces yet. CC0 is the intended metadata for original
generated examples, not a review of external datasets.

Run: python -m vlanix_tiny.logic --output runs/logic-v1 --per-family 256

Expected 1024 examples each split, 3072 total, no duplicate instance IDs.
Never feed all splits into the old random corpus splitter. Tokenizer fitting
uses train only; keep whole question/answer together. Benchmark 2048 vs 4096
and digit isolation before freezing, and check Vietnamese UTF-8 roundtrip
without including Vietnamese in fitting. No model training or GPU in this step.
Future Vietnamese adaptation after 30M is a separate EN+VI research branch,
with English checkpoint preserved and same parameter ceiling. No automatic
tokenizer ID remapping.
