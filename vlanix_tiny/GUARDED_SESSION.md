# P0.9G — persistent homeostasis and hard KL boundary
P0.9F reduced peak drift but missed .025 nats target, with PI saturation.
Add an independent illustrative hard cap .075 nats; no claim of optimality.
Before a trial update, snapshot model, optimizer, growth topology, CPU RNG,
PI state, reference distribution, plan digest, cursor and blocked flag.
After the update, commit to live state only if measured calibration KL is
within cap; otherwise restore the entire pre-update state and block.
Cursor counts accepted batches, never rejected attempts. No automatic retry,
no new growth authorization. Save explicitly to persist current session.
Use local trusted files, one writer; rollback replaces the State object
(never retain external model/optimizer references). Global CPU RNG assumes
single executor; no concurrency or crash-durable transaction guarantee.
No directory fsync, signature, remote RAM, distributed consensus or migration.
Checkpoint identity binds the caller identity and ordered calibration/training
plan digest; it is not cryptographic authentication or a unique organism ID.
Calibration and adaptation are disjoint; KL cannot prove retained knowledge.
Report two real CPU trials and forced rollback tests. Current datasets are
tiny and mathematical accuracy near zero; focus on learning quality before
autonomous growth. No GPU.
Run python -m vlanix_tiny.guarded_session.
