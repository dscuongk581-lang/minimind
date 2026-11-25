# P0.10C — unified state publication
A local durable reference: one SQLite database stores logical replica blocks,
manifests, authority clock, and current pointer. Stage verifies both copies
and full payload digest, then commits storage without advancing state.
Register and publish reverify replicas while holding the same BEGIN IMMEDIATE
transaction as the authority/epoch/version/parent CAS. Read can recover with
one verified replica per block; publishing conservatively requires all three
logical nodes online and both expected replicas per block.
The probe transfers a real guarded-session checkpoint from A to B, rejects
a stale writer, compares next-update decoded checkpoint state, and reads after
one logical node is offline. Unit tests inject corruption, competing CAS,
missing roots, restart and SQL abort; this is not a power-loss test.
A/B are serial local model instances, not remote workers. All logical
replicas share one physical SQLite file and one failure domain; no actual
distributed durability. Caller and DB trusted; hashes are not authentication.
Stage is content-only and does not validate neural semantics or enforce
learning gate/cursor on arbitrary caller data. The probe uses Session.attempt
before publication, but the storage API is not itself a learning-policy gate.
No real-time lease, network, quorum, GC or total storage budget. Failed
publication can leave unreferenced staged blocks. Test cap 10 is permissive
only for parity, not a deployment limit. No real node changes or GPU.
Next: bind candidate state transition to validated learning receipt and
design remote store/authority adapter with signed identity and fencing.
Run python -m vlanix_tiny.state_commit_probe.
