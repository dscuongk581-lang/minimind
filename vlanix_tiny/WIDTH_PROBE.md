# P0.11J — width extrapolation diagnostic
Phase 1 budget stays 5,970,176 parameters. P0.11I run 45575315380
was still in progress when this probe was prepared; no winning weight
selected and no claim about its final performance.
Evaluate every P0.11I arm at the end of its existing training. No extra
optimizer update, no new parameter, no test split use.
64 paired frames per width: 16 validation byte pairs per operator,
then extend those exact low-bit prefixes to 16 and 32 bits with
deterministic pseudorandom high bits. Nested prefixes, same frames across
arms within a seed. All training streams have width 8. Wider probes
are exploratory, not an exhaustive proof; low prefixes are sampled
validation pairs but local transition contexts were supervised.
Expose original operand bit pairs one step at a time, using the
model's previous predicted ID unchanged, with unrestricted argmax.
No final byte requested, no true carry or oracle feedback.
An external decoder reads low bit of each valid predicted state;
it does not repair errors, and is distinct from learned final-byte
output. Compare exact trace, decoded integer modulo 2^width,
invalids, per-position error, first error and prefix parity.
Run a separate gold-prefix diagnostic after each free rollout, never
count it as free accuracy. External scheduler still supplies order
and width; no autonomous planning claim. Wide sequence reaches
98 positions, within model context. Same model/no new learning
isolates inference length extrapolation, not equal inference FLOPs.
Pure Python RED missing module, GREEN four tests passed locally.
Full CPU CI pending. Run python -m vlanix_tiny.output_learning.
