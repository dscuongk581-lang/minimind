# P0.8B — Organic FFN engineering probe
Append a 32-channel SwiGLU branch to a dense FFN; 24,576 parameters at width
256. Original parameter names, object identities and optimizer moments stay
unchanged. Initialize only the new output projection to zero; internal
projections use ordinary initialization. At insertion the branch and its
input Jacobian are zero, preserving the function and old raw gradients.
New output projection can learn immediately; internal projections learn
after a nonzero output update. No gate alpha=0.
Real autograd sensor observes fixed unit projections of each FFN output
gradient by task family. Sum across tokens (CE has mean reduction).
Rank by RMS group means, solely to choose an engineering probe site, NOT
proof of a bottleneck. Within-group cumulant decomposition is diagnostic.
The model is nearly untrained (one warmup update on original train fixture).
No validation/test inputs, no real retention measurement, no autonomous
growth authorization. No change to existing trainer or packed forward.
Only single-process FP32 CPU supported by current tests. No scheduler,
distributed wrappers, global clipping equivalence, checkpoint topology
restore or transactional rollback yet. add_microblock assumes ordinary
AdamW with default settings for appended groups; use only at step boundary.
Next: versioned topology checkpoint and matched validation trial against
base-only update, fixed probes and independent retention set; then PI/PID.
Run python -m vlanix_tiny.organic.
