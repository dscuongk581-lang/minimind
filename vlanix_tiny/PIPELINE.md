# P0.5E — Offline integrated CPU pilot

Input is reviewed local JSONL (e.g. ingestion raw output). The pipeline splits
documents before tokenizer training, freezes tokenizer on train only, chunks
both splits, rejects exact chunk overlap, verifies token counts, generates
paired schedules, trains two models with identical seeds and token budgets,
saves checkpoints, reloads and validates loss. Packing is measured but not
enabled during A/B training. Manifest records hashes and metrics.

```sh
OMP_NUM_THREADS=2 python -m vlanix_tiny.pipeline --input data/wiki-pilot/raw.jsonl --output runs/pilot --budget 128 --batch-size 4
```

Only CPU, FP32, bounded pilot (10 MB CLI input, 2000 steps per arm). Smoke
mode permits reserved placeholder vocabulary solely for synthetic integration
tests. It does not imply a usable language model. Real mode refuses placeholders.
No network ingestion inside this command. Source review and source URL audit
remain prerequisites; preserve original ingestion files separately. Exact
chunk overlap rejection may require upstream corpus refinement. No near-duplicate
guarantee. Checkpoint reload tested, interrupted-resume integration is separate.
Do not commit datasets/checkpoints. CI uses original synthetic text only.
