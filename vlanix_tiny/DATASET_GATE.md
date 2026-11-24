# P0.6C — Cross-budget held-out quarantine

P0.6B actual run 45394670811: 425/43 train/validation documents,
97774/18183 original tokens, full 4096 vocabulary. One exact chunk
collision at budget 64, zero at 128/256/512. Software CI passed 34 tests;
data gate failed.

Keep train and its tokenizer frozen. Quarantine any entire validation
document with an exact chunk collision at ANY of the four budgets,
then compare the same remaining validation documents across budgets.
Record raw collision counts, excluded document and token totals and
post-quarantine metrics. Fail readiness when the retained validation
set is too small or empty. This is a conservative screening choice,
not proof against substring or near-duplicate leakage. The held-out
set is conditioned on overlap; independent untouched evaluation remains
required. No changes to training corpus or source approval. Existing
training pipeline still fails closed on cross-split exact collisions;
integration of the audited split into real training is a later step.
