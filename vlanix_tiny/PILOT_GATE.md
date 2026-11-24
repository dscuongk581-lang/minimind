# P0.6D — audited CPU training gate

P0.6C measured 425 train documents, 42 retained validation documents,
99350/15642 target tokens at context 128. Data gate passed, source
review still pending. Real training requires an operator review receipt
for the exact canonical input JSON rows. This is an attestation, NOT
a cryptographic signature or automatic legal determination.

Review source terms and attribution, retain raw source URLs and review
records privately, then create a JSON receipt with review_id, decision
"approved-internal-research", and input_sha256 computed by review_digest.
Do not fabricate approval. CLI --review-receipt points to the receipt.
Without it the real pipeline refuses training. Smoke remains synthetic
and clearly marked. No source text or model uploaded by CI.

Train tokenizer before auditing, quarantine validation documents on any
exact collision across all four budgets, retain the same train corpus.
Real gate requires >=10 retained validation docs, >=5000 original
validation tokens, 5–20% validation fraction, full vocab, no exact
chunk overlap. This is not a near-duplicate audit. Log per-step training
loss and cumulative tokens; evaluate both arms at the final matched
token budget. Report train-only throughput and Linux process peak RSS.
Packing stays off in the A/B comparison. No GPU required.
