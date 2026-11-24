# P0.5B — Hugging Face ingestion pilot

Only Wikimedia Wikipedia Vietnamese 20231101.vi enabled initially. Dataset card declares CC BY-SA 3.0 and GFDL. Review actual source terms, attribution, article history and share-alike obligations before ingestion; the review ID is an operator reference, not proof of compliance. FineWeb-2 and instruction data are not yet enabled. No data has been downloaded by this commit.

Install datasets==3.6.0 in isolated environment. Supply the actual reviewed 40-character Hub commit SHA (not main).
```sh
python -m vlanix_tiny.ingestion --revision "$REVIEWED_SHA" --review-id "$REVIEW_ID" --output data/wiki-pilot --max-rows 1000 --max-bytes 10000000
python -m vlanix_tiny.corpus --input data/wiki-pilot/raw.jsonl --output data/wiki-clean
```
Limits apply to scanned rows and serialized output bytes, NOT network transfer or cache size. Streaming may fetch larger parquet ranges. Use external network/disk/time quotas for VPS. Pilot collects in memory. Raw manifest and raw records retain source URLs and review references; keep them for attribution because corpus and chunk schema currently retain only core provenance. No dataset or checkpoints committed. Do not train until source review, tokenizer freeze, language/quality gates and held-out split complete. CI uses synthetic rows, no network or secrets.
