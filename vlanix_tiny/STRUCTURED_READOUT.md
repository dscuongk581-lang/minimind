# P0.11Q — structured readout objective
P0.11P CI 45587515532 passed seven jobs and 160 smoke tests.
Canonical/varied accepted 54/263 of 320 batches, unseen carry
59/706 of 8192, arithmetic neural readout 89/262 of 1728.
Both preserved all local contexts and reset traces. Direction
job rerun in P: mixed 186/1728 vs projected 562/1728;
seed42 projected 535 vs mixed33, seed43 projected27 vs mixed153.
Not a stable improvement. No projection used here.
Use P varied dataset, same seed/order/foundation, 160 proposed
batches, adaptive exact guard and frozen tied E. Compare CE vs
CE + 0.75*8*mean bit marginal NLL on 24 readout targets only.
Bit marginal conditioned on byte IDs 9..264, full CE still
penalizes control IDs; local rehearsal CE unchanged. No new
parameters or output head. Lambda8 exploratory. Report
unconstrained full argmax primary, separate marginal bit
decoder diagnostic on canonical and held-out carry probes,
bit accuracy, local retention and arithmetic readout.
Marginal decoder may emit a byte even if a control ID wins;
never use as corrected primary score. All bytes seen in train,
held-out carry masks only. Local RED missing module then GREEN
4 tests in 0.201s with CPU torch. Full model CI pending.
No test split, GPU, growth or deployment.
