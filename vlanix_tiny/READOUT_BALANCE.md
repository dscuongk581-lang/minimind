# P0.11P — balanced neural readout
P0.11N passed CI 45584808525. Adaptive readout 483/1728 with
273/320 accepted batches and 32/32 local throughout.
P0.11O direction job was still running at implementation; no
projection advantage assumed. Independent diagnostic of byte readout.
Both arms reproduce 120-step local foundation, then use same 160
proposed batches, 3840 readout examples, 1280 local rehearsal examples,
AdamW state, RNG, adaptive guard and frozen E. Each byte 0..255 appears
15 times in the same target order. Canonical always uses carry=0;
varied uses carry=0 plus 14 distinct random masks per byte. Probe
4096 nonzero masks (16 per byte), disjoint from varied train frames
and canonical frames. All target bytes are seen, so this is invariance
to nuisance carry, not generalization to unseen byte values. Some
synthetic carry sequences may not be physically reachable by ADD;
they test the readout interface independent of arithmetic. The 256
zero-mask probe is in training, explicitly not held-out. Also evaluate
on existing arithmetic validation via cached learned local transition
table, verifying single-call parity. Neural argmax must output a byte
without external correction. No oracle used for free arithmetic traces.
No held-out arithmetic test split. Compare same proposed budget,
not compute or accepted budget (adaptive retries may differ).
No new token, shape or parameter. Local RED import missing then
GREEN 4/4 tests in 0.098 s (Python 3.13). Model CPU CI pending.
