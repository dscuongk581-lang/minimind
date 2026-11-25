# P0.9A — VLANIX Binary Native (research reference)
Separate from the existing 4096-BPE lineage. No conversion of trained
weights or change to MiniMind. Protocol version 1 reserves IDs 0..8 for the
existing control IDs (PAD/BOS/EOS/UNK/system/user/assistant/tool_call/
tool_response). Raw bytes map to IDs 9..264, losslessly, without normalization.
The strict frame decoder accepts only BOS, zero or more byte IDs, EOS;
other controls are not accepted as payload. IDs are u16, payload bytes u8.
No execution of input bytes. Context-length limits and chunking belong to a
later protocol layer. Byte-level is a candidate, not a proven training win.

One row-major E[265,256] f32 allocation: lookup E[id], output E*h.
67,840 parameters; 271,360 bytes of weights. Output is logits, not an inverse.
Snapshot layout: ASCII VLXEMB01 (8 bytes), vocab u32 LE, width u32 LE,
then finite f32 LE in row-major order. Exactly 271,376 bytes. No addresses
or platform-native structs serialized. Reject malformed length, header,
shape, non-finite values; NOT a cryptographic integrity check.
File digest, authenticated topology checkpoint, optimizer and RNG are future
work; REA does not understand this custom format without a parser.
REA is an external read-only artifact inspector (GGUF/Safetensors, compiled
Rust artifacts with cargo-auditable metadata). No REA runtime dependency.

CPU scalar arithmetic here is a reference executor, not in-memory physical
compute. Portable state does not guarantee bitwise equivalence across future
kernels. No learning/autograd, PID, VCPU, GPU or distributed RAM integration.
Next: independent Python parity and binary corpus benchmark, byte model
training, versioned topology/optimizer snapshot. Keep the 30M cap.
Run: cargo test --manifest-path vlanix_binary/Cargo.toml --locked --offline.
