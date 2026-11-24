# P0.5A — VLANIX Chunk V1

Boundary-aware (heuristic, not semantic-model based), lossless segmentation.
64/128/256/512 total tokens including BOS/EOS. No overlap or silent truncation.
Paragraph then sentence then whitespace preference when a boundary fits; character
fallback for long unbroken text. Exact re-encoding and round-trip checks.
Offsets are Python Unicode code-point positions, not UTF-8 byte offsets.
Chunk ID binds source, document, document hash, tokenizer serialization hash,
budget, offsets and split. Input must be reviewed NFC corpus already split by
document; run separately on train and validation, never merge before chunking.
Tokenizer must be frozen before production chunks. Smoke tokenizer for tests only.

```sh
python -m vlanix_tiny.chunks --input data/train.jsonl --output data/train-128.jsonl --tokenizer data/tokenizer.json --split train --budget 128
bash vlanix_tiny/ci.sh
```

No external corpus imported yet. Near-duplicate detection, packing, curriculum
and actual source ingestion are later stages. Chunk length is not a proxy for
semantic difficulty. Existing trainer accepts text JSONL; use matching length
and avoid extra splitting. Compare budgets on equal target-token counts.
