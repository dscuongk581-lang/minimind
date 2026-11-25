# P0.10B — local executor authority
One SQLite authority, serialized BEGIN IMMEDIATE transactions and synchronous
FULL. Persist identity, manifest root, version, current owner, fencing epoch,
logical expiry, and authority clock. Transfer or re-acquire after expiry
increments epoch; commit compares owner/epoch/expiry/version/parent root in
the same transaction. Rejected writes cannot update the canonical pointer.
Logical time advances only via trusted authority advance(), not real time.
Expiry is not automatic; no heartbeat or renewal, no wall-clock guarantee.
Tests cover conflicting acquisition via independent connections, transfer,
stale writer, replay, parent mismatch, re-open, identity isolation.
All callers and local database trusted. Owner names are not authentication;
no network, signing, quorum, failover authority or cross-node consensus.
This registry only validates digest syntax and CAS, not block availability,
checkpoint correctness or learning quality. No coupling to VirtualRAM yet.
Next: verify and stage candidate checkpoint in replicated store, then CAS
manifest publication; test stale worker with real checkpoint and crash
boundaries before live deployment. No production node changes or GPU.
Run python -m unittest discover -s tests -p 'test_tiny_executor_lease.py' -v.
