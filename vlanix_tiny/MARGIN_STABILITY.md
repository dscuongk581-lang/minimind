# P0.11S — stable knowledge learning
R run 45600741572 complete-rehearsal job and 168 smoke tests PASS.
Overall workflow cancelled because gradient-direction job cancelled.
Sampled vs complete: neural arithmetic readout 798/1431 of 1728;
held carry 3538/6516 of 8192; accepted 271/310 of 320; attempts
794/539. Minimum committed margins across seeds .00360584 vs
.11778831. Complete preferred for next experiment, not a universal
claim. Both kept all 32 local argmax predictions.
S compares full rehearsal CE versus same CE plus .25 mean
ReLU(1 - margin) on all 32 local contexts. Threshold and weight
exploratory. This is a soft training penalty, NOT a new hard
acceptance threshold. Both retain exact 32-context adaptive guard.
Same 56x10 batches, 24 readout + 32 local, varied carry schedule,
160 proposed batches, 120 foundation steps, frozen E, optimizer/RNG.
Predeclared seeds 42,43,44,45, separate CI matrix jobs. A failed
foundation is explicitly reported, not silently excluded or retrained.
Report seed-level results before aggregation, accepted attempts,
committed margin trace, unseen carry and arithmetic neural readout.
All bytes seen in training; no arithmetic test split used. No
claim equal compute (retry counts may differ), monotone margins,
or distribution preservation. No parameter, vocabulary or GPU change.
Local actual RED import missing, GREEN 4 unittest tests in 0.017s
torch2.10 CPU. Full pinned torch2.6/transformers model CI pending.
Run python -m vlanix_tiny.margin_stability_learning --seed 42.
