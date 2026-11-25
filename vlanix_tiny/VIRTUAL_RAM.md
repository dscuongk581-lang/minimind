# P0.10A — virtual RAM migration reference
64 KiB immutable SHA-256 addressed chunks, ordered manifest with full
payload digest and caller-pinned manifest root. Three in-process maps
simulate logical nodes; each unique chunk has two deterministic replicas.
Read verifies every chunk and fails closed when no valid replica remains.
Dedup by content, no encryption, authentication, quota per node or GC.
Manifest root must come from a trusted channel; hashes do not grant authority.
512 MiB per payload limit, not a bounded lifetime memory pool.
Replicas can share Python bytes objects: not independent physical RAM.
Fresh CPU subprocess restores the complete session after simulated loss
of one node and performs the next identical update. Compare all decoded
checkpoint fields including model, optimizer, PI, reference, RNG, cursor.
Byte parity applies to the reconstructed input file; decoded state equality
applies to the subsequent checkpoint (archive metadata can vary).
The caller passes the ordered plan separately. Test cap 10 nats is deliberately
permissive for continuation, NOT an approved deployment limit.
No real OptiBlockAI or VCPU call, network, failover lease, durable storage,
split-brain prevention or cross-hardware parity. Source and worker are
serial comparison branches, not concurrent writers.
Next: define read-only integration contract for OptiBlockAI; external
worker lease/epoch fencing before any live migration. Parallel track:
improve binary curriculum; no GPU or automatic growth authorization.
Run python -m vlanix_tiny.ram_migration.
