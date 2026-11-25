# P0.9F — mathematical homeostasis
P0.9E verified state restore and continuation but did not beat fixed capacity.
Switch all arms here to query-only answer CE (no EOS supervision), so EOS
learning cannot masquerade as learning a mathematical answer. Warmup on
48 train examples per old family; reserve 16 per family from the train split
for feedback, never optimize their labels. New-family adaptation uses 64
train examples per family. Validation is measurement only, not feedback;
test unused for training or metrics. Two seeds, 42 and 43.
Measure KL(reference || current) over all 265 output IDs on 48 calibration
queries. Reference is the post-warmup model, not a competent teacher.
Use PI on filtered KL: threshold .025 nats, kp4, ki1, smoothing .5,
base-LR multiplier in [.05,1], conditional anti-windup. New block LR remains
.001. Fixed, growth without PI, and growth with PI use same warm checkpoint,
same adaptation rows and steps; growth arms use same seeded block.
This is matched steps, not FLOPs; all gradients globally clipped at 1.
Report old/new validation answer CE/accuracy and full calibration drift
trace. KL is distribution preservation, not a proof of knowledge retention.
No production growth approval, no PID derivative yet, no cross-node state
or persistent controller checkpoint. Gains and threshold are uncalibrated.
Use measured tradeoffs before tuning; do not tune on test.
Run python -m vlanix_tiny.binary_homeostasis.
