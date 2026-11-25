# P0.11I — mathematical output and local knowledge retention
P0.11H CI 45572974464: 128 tests passed. Stream exact trace
1621/1728 vs front 322/1728; free final-byte accuracy 93/1728
vs 126/1728. Local table retention stream 46/64 vs front 61/64.
E exactly fixed. Do not conflate correct trace with correct byte.
Repeat P0.11H stream data, initialization, foundation, AdamW state,
CPU RNG, schedule, 160 updates, 3840 trace examples and 1280
local examples per arm. Each batch has 192 state, 24 final and 8
local target positions. Factorial per-token weights (state, final,
local): (1,1,1), (1,8,1), (1,1,8), (1,8,8). Normalize by sum of
weights; all groups still contribute, same steps and supervision.
No new parameters. No claim that 8 is optimal. Baseline exactly
uses original token CE. Weighted objective values not directly
comparable across arms. Measure unconstrained free trace and final
byte, joint accuracy, consistency, invalids and local table
retention. Report deterministic external decode of predicted trace
and final-byte prediction given true trace as two separate diagnostics;
neither corrects or participates in free rollout. No test metrics.
Two seeds; exploratory. Existing smoke job unchanged except optional
loss hook, new output-objective job runs separately to avoid stacking
all training under one timeout. No GPU, growth or runtime deployment.
Local CPU PyTorch 2.10: RED missing module, GREEN four tests passed;
full repo CI uses pinned torch 2.6 and transformers 4.57.6.
Run python -m vlanix_tiny.output_learning.
