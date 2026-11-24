# P0.5D — Curriculum experiment V1

Order-only comparison: same frozen chunks, same target-token totals, same
number of optimizer steps, random baseline versus ascending token-length
order within each epoch. Seeded shuffle provides deterministic tie-breaking.
No validation chunks allowed. Partial final batches are preserved.
Length is only a proxy; no claim of semantic difficulty or improved learning.

Use fixed padding context, same initial weights, batch size, optimizer,
learning rate and held-out evaluation for both arms. Batch token counts
may differ; use token-weighted reporting, and treat batch composition as
part of the intervention. Do not enable packing for this first comparison.
Repeat at least three paired seeds before interpreting results.
Evaluate at matched total target-token exposure, not arbitrary step numbers.

```sh
python -m vlanix_tiny.curriculum --input data/train-512.jsonl --output runs/schedules --epochs 1 --batch-size 4
bash vlanix_tiny/ci.sh
```
The CLI creates schedules and hashes, not trained models. Existing trainer
still samples randomly; schedule-driven execution and end-to-end dataset
validation are next milestones. No GPU required for this stage.
