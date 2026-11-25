# P0.10D — learning receipts
Local trusted CPU reference. Genesis pins session plan, checkpoint identity,
cursor zero and exact research KL cap .075, and checks reference drift.
Genesis is trusted, not proof of its prior training history.
For each publication reconstruct the parent and candidate, replay exactly
one guarded answer-learning step from parent, reject if gate rejects or any
decoded checkpoint field differs. Restore caller CPU RNG after replay.
Compare cursor with version. Recheck authority, parent CAS and replica
integrity at publication; atomically insert hash-linked receipt and update
canonical pointer. Receipt includes batch and plan digests, measured event,
owner, epoch, parent and new roots, version and prior receipt digest.
Audit checks chain and current pointer; not a replay of all historical
learning. Rejected candidates are not published; rejected attempts and
blocked state are not durably audited in this version. No auto retry.
LearningStore.commit rejects ordinary writes; the underlying StateStore
and direct DB access are trusted and can bypass policy, not security
boundaries. No signatures or external audit anchor; a privileged DB writer
can rewrite history. Same-host deterministic replay is expensive, not a
proof of arbitrary worker computations, cross-hardware parity or neural
competence. Only trusted local archives, single executor/threaded
environment with serialized learning and global RNG; no hostile checkpoint
sandbox, external identity, remote execution or deployment. No GPU.
Probe checks forged weight, stale owner, direct commit rejection, valid
replay, real KL rejection, database reopen and audit tamper detection.
Run python -m vlanix_tiny.learning_receipt_probe.
