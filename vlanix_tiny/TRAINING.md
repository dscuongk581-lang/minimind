# P0.4 — Training baseline

Single device CPU/CUDA FP32, random weights, AdamW, gradient clipping, token-weighted validation, deterministic sampling, atomic checkpoint replacement, optimizer/RNG resume and JSONL metrics. In-memory corpus encoding for small pilots only. CUDA is implemented but not validated until a GPU is available.

Run after P0.5 delivers a reviewed corpus and a fully learned tokenizer with no reserved placeholders:
```sh
OMP_NUM_THREADS=2 python -m vlanix_tiny.training --train data/train.jsonl --validation data/validation.jsonl --tokenizer data/tokenizer.json --output runs/pilot --steps 20 --device cpu
```
Resume with identical arguments, larger --steps and --resume. Use trusted local checkpoints only. Outputs include state.pt and metrics.jsonl; do not commit them. Reported token throughput includes evaluation/checkpoint overhead; RSS units are Linux KiB. Checkpoint tests use synthetic token IDs and do not demonstrate Vietnamese fluency.

CI: bash vlanix_tiny/ci.sh. GPU rental gate remains closed pending real corpus, tokenizer, CPU pilot, recovery, evaluation and GPU packaging.
