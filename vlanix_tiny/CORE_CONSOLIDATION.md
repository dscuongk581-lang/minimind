# P0.11X — consolidate the finite core
V run 45614515604 and W run 45615004505: all four seeds
have exact 262144 arithmetic cases and 832 test cases per
seed; 43,44,45 have exact readout 65536, seed42 65529/65536.
One scheduled and six other readout frames wrong; all held
4096 correct. W is blocked by V's strict gate. No quota claim.
Use exact archived checkpoints from W run 45615004505,
all four available at implementation, SHA-256 pinned in code.
Allow a failed finite readout gate only for consolidation,
never for SUB; require parent local and arithmetic exact.
Third fixed 160-batch pass, 3840 new carry frames disjoint
from both preceding training schedules and held masks.
All seeds treated alike, no per-error training or retuning.
Full rehearsal, margin, adaptive guard, same optimizer/RNG
and E as parent. Original checkpoints untouched. Verify
65536 readout and 262144 arithmetic cases again, with 32
local rules and 832 previously opened test rows. Test now
is regression, not independent evidence. Exhaustive gate
unchanged (explicit local correctness also required).
Third schedule changes coverage grouping to 11520 scheduled,
4096 held, 49920 other. Scheduled does not imply accepted.
New candidates are saved separately, hash recorded, no promotion
or SUB in automatic CI until separately reviewed. Old V/W
jobs remain manual; on push smoke and X matrix only.
Cross-run artifact download uses read-only actions permission
and fixed run ID; artifact expires after seven days.
Actual local RED import missing then GREEN 4 tests in 0.081s.
SHA256 core_consolidation.py f8fc110f4db582359a112ea12e119395e06ed21ac27e29c24aceb794534cbcb0
SHA256 test_tiny_core_consolidation.py f8b52ddf54fc82636c5ecb2e8e3c3956cf746f644fe5c8fb81193fbda11b6f7f
Full model CI pending. No GPU, vocabulary or parameter growth.
