# P0.11T — learning-state decoupling
S run 45603704843 all 4 margin jobs success; workflow cancelled,
smoke log 172 tests, not full workflow pass. Complete/margin
arithmetic byte 2223/2460 of 3456; held carry 10027/11314 of 16384;
accepted 497/508 of 640; seed43 both 27/864 despite initial
margin 2.85 and minimum committed margin .833. Seed43 is not
stable across separate CI runs. No claim of causal mechanism.
Test whether optimizer history contributes to task-boundary stall.
Both arms start from identical 120-step local foundation weights,
same LR/AdamW hyperparameters, data order, CPU RNG, complete
rehearsal plus .25 soft hinge at margin1, adaptive guard, 160
proposed batches, 42/43/44/45 seeds. Retain all AdamW slots vs
clear all slots once at task boundary (both moments and step).
This is joint optimizer-state intervention, not isolated momentum.
No weight or token change. Both validate 32/32 before and after
commits. Compare per-seed results and failure explicitly. Retry
compute may differ. No hold-out arithmetic test use.
To reduce repeated costly experiments and cancellation risk, older
research jobs remain available via workflow_dispatch; on push retain
full existing smoke regression and latest four-seed experiment.
No claims that cancellations are quota failures.
Local actual RED import missing, GREEN 4/4 unittest 1.108s
torch2.10.0+cpu. SHA256 learning_state.py
715ed8ef59016be0ccf31fe4000f3b856057f5308614aaa1d6716b4b401561fa
tests/test_tiny_learning_state.py
50d501ffe0bf1e555eea44693b1a8aeb7ed861fd1c184b9b90f7c2c1c278848a
Full pinned CPU CI pending. No growth, GPU or deployment.
