# P0.6A — Vietnamese Wikipedia source preflight

Preliminary agent screening, not human license approval. Official source:
https://huggingface.co/datasets/wikimedia/wikipedia
Pinned revision adding Vietnamese files:
https://huggingface.co/datasets/wikimedia/wikipedia/commit/859a94672c6c0a0cfd1c19c5506d02e0e4604cda
Dataset card declares CC BY-SA 3.0 and GFDL; source terms and attribution
requirements must be reviewed before real model training/distribution.
Schema id/url/title/text confirmed using Dataset Viewer.

Preflight streams only first pinned Parquet shard, scans <=128 rows by
default and retains <=1 MB serialized records in memory. This is NOT a
network transfer ceiling. CI timeout 10 min, CPU only, no credentials.
Split before tokenizer learning; report vocabulary placeholders, chunk
counts, token counts, exact overlap and rejection counts. No raw text,
tokenizer or weights uploaded or committed. No real model training yet.
A contiguous first-shard sample is not representative of Vietnamese.

If preflight succeeds, review source provenance and licensing, refine
quality and split if needed, then run real CPU A/B pilot with reviewed
ingestion. Keep the original human review requirement for ingestion CLI.
Fallback on VPS: same command in isolated venv with two threads, a disk
quota and timeout; do not move code failures to another CI.
