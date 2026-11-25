# P0.11N — adaptive protected learning
P0.11M protected-core job 139096057870 PASS: ordinary 1188/1728
neural bytes, guarded 68/1728, both end 64/64 local transitions
and 128/128 reset 32-bit traces; guarded accepted only 10/320
batches, ordinary temporarily lost local accuracy (min14/32).
Full M smoke was still running at implementation.
Retry each proposed batch at LR scales 1,1/4,1/16,1/64.
Each rejected attempt restores weights, optimizer moments, CPU RNG
and training mode via existing guard, with all base LRs restored.
Accept first candidate passing all 32 local contexts; otherwise skip.
Frozen E stays zero LR. No change to guard or model size.
Same deterministic two-seed foundation, batch order, data and
160 proposed batches as M. Compare single-try guard vs adaptive.
Report accepted batches, attempts, accepted scale counts/sum, target
tokens including retries, neural byte and reset accuracy, timing.
Not matched compute: up to 4 tries/batch. No claim monotone
loss, convergence or global knowledge preservation. Small accepted
steps may still stall. Guard uses known training truth table only.
Validation not used for acceptance or selection. No test split.
Actual local RED missing module then GREEN 4 tests (0.695 s).
Full CPU CI required. No GPU, growth or deployment.
