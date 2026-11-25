# P0.11U — consolidation
T run 45606890050 four learning-state jobs passed; smoke pending
when U started. Retain/reset arithmetic readout 3045/2968 of 3456,
held carry 13907/13917 of 16384; both accepted 637/640, attempts
1030/936. No consistent reset advantage; retain for U.
All four seeds 32/32 throughout. Cross-run variation remains.
U predeclares seeds42..45, same local foundation120 steps and
common first readout stage160 batches (retain, complete rehearsal,
soft margin .25, target1, adaptive exact guard, frozen E).
Measure stage1 then fork identical weights, AdamW state and RNG.
Two second stages of 160 proposed batches with identical target
byte order and sampled local schedule, complete rehearsal.
Repeat uses original 3840 frames; Fresh uses 3840 new distinct
(byte,carry) pairs, excluding both original train and held masks.
Each byte appears 15 times in both. Held masks used only to
exclude overlap; no score-driven training decisions. Both total
320 proposed readout batches, retry compute may differ. Report
before/after held carry and arithmetic validation, local accuracy,
margins and accepted steps. All byte labels seen; not unseen-byte
generalization. No arithmetic test split used. No winner declared
until CI. Older research jobs remain workflow_dispatch only; smoke
and latest matrix auto-run. No new parameters or GPU.
Actual local RED missing module -> GREEN 4 tests in 0.022s.
SHA256 consolidation.py edf477d8fd93d1d37aa1bacc5495182319fe784216df6781e70acc90148cac43
SHA256 test_tiny_consolidation.py b7093a322abea99123d04d3b883d152b4dda1f30427619af3a9fb08cb2098652
