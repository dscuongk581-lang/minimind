# P0.3 — Corpus gate

Offline, standard library only. Each input JSONL record requires text, document_id, source and license. License allowlist is a metadata gate, not proof of rights: review attribution, provenance, privacy and applicable share-alike obligations before ingestion. Preserve original source attribution separately. No external dataset imported in P0.3.

NFC normalization, exact deduplication, connected document groups for duplicate texts, deterministic document-level train/validation split, reserved marker and control character rejection, basic email screening, SHA256 output manifest. No guarantee of near-duplicate or PII removal; language identification and manual quality review remain next gates. In-memory implementation for small pilots only.

Run:
```sh
python -m vlanix_tiny.corpus --input /path/to/reviewed.jsonl --output /path/to/new-output
python -m vlanix_tiny.tokenizer --corpus /path/to/new-output/train.jsonl --output /path/to/tokenizer.json
bash vlanix_tiny/ci.sh
```

Train tokenizer only on train split. Never train on validation. Do not commit private data or checkpoints. Smoke fixtures are not a real pretraining corpus.

## CI portability
Same ci.sh on GitHub and VPS 105, in an isolated Python 3.11 virtualenv with torch 2.6.0 CPU and transformers 4.57.6 installed. Checkout the exact approved commit; run unprivileged with CPU/RAM/time limits. No secrets required by tests. VPS integration is not deployed by this commit.
