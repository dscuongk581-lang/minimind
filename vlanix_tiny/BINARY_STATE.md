# P0.9E — local state continuity and growth trial
Trusted local CPU checkpoint: version, dataset identity, chronological
MicroBlock topology, state_dict, AdamW state, step counter, CPU RNG.
Restore topology in insertion order before loading optimizer. Save via
temporary file + fsync + replace; directory fsync, cryptographic authentication,
distributed replication, data cursor and cross-platform bit parity not provided.
No claim of crash-durable distributed checkpoint. No untrusted checkpoint input.
Train warmup on arithmetic/boolean/state; adapt pattern/inference/recurrence.
Same initial checkpoint, same adaptation batches and steps, fixed vs one
32-neuron block at sensor-selected layer. Masked answer/EOS gradients drive
sensor. Compare query-only answer CE and accuracy on old/new validation.
Same steps/target tokens, NOT exact FLOPs. Single seed, small synthetic OOD
fixture. No test metrics, no GPU, no autonomous authorization. Measure
insertion identity, restore identity, next-update exact equality on CPU.
Thresholds and PI/PID follow after observing repeated trials.
Run python -m vlanix_tiny.binary_growth.
