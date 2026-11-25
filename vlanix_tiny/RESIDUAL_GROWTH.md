# P0.8A — residual-driven growth research

The theta P32 identity motivates a diagnostic, not a transfer of its asymptotic
constants. For a fixed projected gradient q and task group D, center within
each group: R=q-E[q|D]. With sample-count weights, the empirical population
cumulant obeys k4(R)=E[k4(R|D)]+3 Var(Var(R|D)). Also report the between-group
mean variance separately so centering does not hide coherent signals.
These are empirical moments, not unbiased population estimates. Small
samples, outliers and choice of projection can dominate them.

A scalar squared-loss toy probe compares coherent and cancelling gradients
with equal starting loss. It is deliberately constructed with an unaffected
retention feature (z=0) and repeated support, not independent generalization.
A toy gate requires three same-sign projected gradients, positive validation
gain, bounded retention delta and total count <=30M. Thresholds .05, .001,
.01 are illustrative and uncalibrated; units depend on loss and projection.
PROBE_ACCEPT does NOT allocate any weights. No model or optimizer is mutated.
Do not use kurtosis alone to decide growth. P0.8B: measure real autograd
projections, independent batches, compare candidate MicroBlock probe vs
matched old-parameter training, then conservation and checkpoint gates.
Compare fixed growth, gradient-only and residual-aware approaches; use
PI/PID only after measuring plant response. Machine-language-first remains
a research arm, not a frozen tokenizer. Fixed embedding dimensions and tied
output interface; weight freezing after foundation learning is experimental.
No external corpus, GPU rental, or 6M training in this step.

Run: python -m vlanix_tiny.residual
Tests: python -m unittest discover -s tests -p 'test_tiny_residual.py' -v
