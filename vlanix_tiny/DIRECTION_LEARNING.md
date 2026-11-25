# P0.11O — gradient direction conservation
P0.11N adaptive-core job 139103577442 passed. Across two seeds,
single-try guard accepted 8/320 batches, adaptive 273/320;
neural readout 60/1728 vs 483/1728; both kept 32/32 local
contexts throughout and 128/128 exact reset traces at width32.
Adaptive required 762 attempts. Overall N pipeline still running.
New matched arms: same foundation, work order, 160 proposed
batches per seed, 24 readout and 8 rehearsal samples, optimizer
moments and adaptive guard. Both compute separate CE gradients
g (readout) and h (local rehearsal) over core parameters only.
If dot(g,h)<0, project g to g-dot(g,h)/||h||^2*h.
Compare ordinary .75g+.25h to .75 projected_g+.25h.
Projection is global across tensors, not per layer. Both exclude
tied frozen E from gradient clipping and subsequent AdamW moment
updates, so neither is an exact reproduction of N. E remains fixed.
Measure conflict count per attempted update, accepted scales,
neural readout, local retention and reset width traces.
First-order gradient property only; AdamW preconditioning and
finite step can still violate the constraint, so exact postcheck
and rollback remain. Rehearsal gradient from eight sampled
contexts is not a guarantee for all 32. Retries may differ in
compute. No claims of general knowledge preservation.
Actual local RED missing module, GREEN 4/4 unittest 0.005 s.
Full model CPU CI pending, no test split, growth or GPU.
