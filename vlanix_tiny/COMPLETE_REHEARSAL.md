# P0.11R — complete local-rule rehearsal
P0.11Q structured-readout job 139151346880 passed; overall Q
pipeline running when R started. Across 2 seeds CE vs CE+Bit8:
unseen carry 1199/120 of 8192, bit correct 46550/43437 of 65536,
arithmetic neural readout 321/98 of 1728, accepted 178/97 of 320.
Both preserve 32/32 local contexts at every committed step.
Bit8 not selected. Hypothesis: sampled rehearsal may fail to
protect the contexts not seen in a given batch.
Use same varied carry plan and probe, seeds42/43, frozen E,
120-step local foundation, AdamW state/RNG, 160 proposed batches.
Both arms use 24 readout rows plus 32 local rows, padded length10.
Sampled repeats the same sampled 8 contexts four times; Complete
uses each of all 32 once. Loss .75*mean readout CE + .25*mean
local CE keeps the relative objective weights fixed. Duplicate
control is equivalent in exact arithmetic to prior 24+8 CE,
but larger batch shape may alter numerical execution.
Adaptive guard and four LR scales unchanged. Record exact local
argmax at every committed batch, minimum true logit minus max
competing logit, mean margin every 20 batches, accepted attempts,
retry token counts, same arithmetic validation and held carry
probe. Margins are diagnostics, not additional guard thresholds.
Full rehearsal not a guarantee of better learning. Same shape
and proposed data budget, retries can differ in compute.
All byte values seen, held masks only; no test split used.
Local RED import missing then GREEN 4 tests in 0.074s CPU.
Full model CI pending. No new parameter, GPU or deployment.
