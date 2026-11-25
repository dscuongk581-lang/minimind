# P0.11V — finite mathematical core verification
U run 45608449583 all five active jobs success, 180 smoke tests.
Both final arms 3456/3456 arithmetic validation and 16384/16384
held carry across four seeds; Fresh second pass 640/640 attempts
vs Repeat 648/640. Select Fresh recipe now, before test scoring.
Re-train fixed recipe (120 foundation, 160+160 proposed readout
batches, same seeds and data, full rehearsal, margin, adaptive
guard, frozen E, retain AdamW). U weights were not archived;
these are reconstructed candidates, not identical checkpoints.
Save model, optimizer, CPU RNG and hash before independent
arithmetic test split scoring; no subsequent training. Four
matrix seeds, no seed selection, exact gate per seed.
Enumerate all 4^8=65536 packed-state sequences, run actual model
265-way argmax once per sequence in batches of64. Score scheduled
7680, held4096 and other53760 combinations separately (scheduled
is not necessarily accepted/exposed if a batch was rejected).
Then enumerate 4*256^2=262144 ordered arithmetic cases via
cached model local transitions and cached neural readout IDs.
No oracle feedback or external byte correction. Expected labels
used only for scoring. Also use original 832 arithmetic test
rows and direct neural readout with model-generated traces;
report aggregate agreement with cached evaluation, not per-row
batch-size parity. All byte labels and 32 local frames were
supervised; this is exhaustive finite-domain verification, not
unseen operator or autonomous reasoning. Test operand pairs
were not used to optimize the model; readout supervision can
overlap their intermediate representations. Finite gate requires
all exact; a failure is a research outcome, not a quota failure.
CI artifacts contain candidate checkpoint/report (7 days, not
durable deployment). No parameters, vocabulary or GPU growth.
Local actual RED import missing then GREEN 4/4 in 0.136s.
core_verification.py SHA256 7076df8561f46ddf0ef671fe40bec6124c29b4a59d89d9813c8d97f29f1a9623
test_tiny_core_verification.py SHA256 200ff0e2c8c007310dd8ab859a5dc9ad277634916da2d7bd30489b86d131b8fd
Full model CI pending.
