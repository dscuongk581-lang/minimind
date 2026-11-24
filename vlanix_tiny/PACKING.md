# P0.5C — Packing reference

Deterministic next-fit packing of complete BOS/content/EOS sequences, no
truncation. Fixed context 3..512. Segment-local causal attention, RoPE reset
per segment, BOS and padding excluded from target loss. Separate reference
forward path for Dense Tiny; upstream model remains untouched. Tests compare
loss and gradients with independent sequence training, and check density and
invalid inputs. No production trainer integration yet; compare throughput
and memory before enabling it. Dense block mask may reduce kernel efficiency;
padding reduction does not guarantee speedup. Use frozen tokenizer and
separate train/validation chunks; never mix splits.

CI: bash vlanix_tiny/ci.sh
