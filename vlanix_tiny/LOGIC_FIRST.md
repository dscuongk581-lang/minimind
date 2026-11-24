# VLANIX Tiny NGC — research direction (2026-11-24)
User direction: logic-first 6M, incremental function-preserving growth,
final parameter ceiling 30,000,000. No GPU rental until readiness gate.
The current Wikipedia pilot remains a language baseline, not the foundation
training corpus. No dataset is approved automatically by this document.

## Sources screened
See logic_sources.json. Metadata from Hugging Face, source cards and original
RuleTaker repository. License labels are not proof of underlying rights.
Pin immutable revisions, verify original license and attribution, review
translations, reject invalid labels/solutions, keep test splits and derived
problem families out of tokenizer fitting and training. No bulk download.
Procedural Pile (24.8M rows, 55 families, 13.4 GB train Parquet) and Reasoning
Gym are generator references; use selected low-depth tasks only.
Generate original Vietnamese arithmetic, Boolean logic, rule closure,
ordering and state transitions with deterministic verifiers, and split by
problem family/template, canonical symbolic instance and seed before rendering.
Hold out novel combinations, operand ranges and wording. Keep unknown distinct
from false under an explicitly declared open-world semantics.

## Candidate training mix, by actual target tokens
40% verified deduction/Boolean/relations; 30% arithmetic and patterns;
15% state transitions and simple algorithms; 15% clean Vietnamese prose.
Compare with a prose-rich matched baseline. These proportions are hypotheses.
Each problem has a canonical representation, verified answer, provenance,
difficulty and version; preserve complete premise/question/answer/proof units.
Long tasks are deferred or rendered as self-contained subproblems, never
blindly cut through proof/answer. Existing Wikipedia chunk protocol is a
baseline, not yet the final logic protocol.

## Tokenizer candidates before freezing
A: existing 4096 ByteLevel BPE, train on mixed training-only text.
B: same 4096 capacity, byte-level BPE with digit isolation, preserving operators
and round-trip. C: 4096 BPE with explicit numerical grouping as exploratory
alternative. No candidate gets extra model parameters.
Measure NFC round-trip, unknown count, reserved count, Vietnamese syllable
fertility, tokens per byte and per verified problem, numeric segmentation,
whole-problem fit rates at 64/128/256/512, held-out exact answer accuracy,
training FLOPs/time. Prefer reasoning accuracy over raw compression.
Freeze tokenizer artifact SHA, normalization, special IDs, pretokenizer,
BPE merges, decoder, and versioned chunk/packing/loss protocol only after
selection. Test fixtures are independent from tokenizer fitting.
4096*256 = 1048576 tied embedding weights (~17.3% of 6M); increasing vocab
by 4096 would consume another 1048576 weights at same width.
Paper: https://arxiv.org/abs/2402.14903

## Growth (candidate, not benchmarked)
Fixed width 256, FFN 832, 4 Q / 2 KV heads, vocab 4096.
Parameter formula from current preset: 1048832 + 836224 * depth.
6 layers = 6066176, 8 = 7738624, 12 = 11083520, 18 = 16100864,
24 = 21118208, 34 = 29480448. Depth-only schedule is an upper-bound
candidate, not a claim of optimal depth. Alternative final design:
24 layers, FFN 1312, estimated 29965568 weights; expand FFN by copying
old channels and zeroing new down-projection columns. Verify counts in code.
Growth gate: logits, old gradients (same inputs/loss, before clipping),
optimizer state by parameter name, learnability of new weights, retention
and new-task performance. Zero output projections in inserted pre-norm
residual blocks, normal initialization inside. Group clipping and LR
tested separately; don't claim full training trajectory equivalence.
Compare continued 6M, grown model, scratch model under equal data and
separately equal compute. Compression cycle optional after evidence.
No growth or logic training has been implemented by this research record.
