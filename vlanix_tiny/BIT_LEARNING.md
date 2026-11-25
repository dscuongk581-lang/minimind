# P0.11D — bit-level mathematical learning
P0.11C 107 tests PASS. Frozen/replay learned OR 117/432 vs
6/432 before adaptation, retained old 89/1296 vs 76/1296 before.
Exact old skill remains weak; prioritize objective, not expansion.
Given 265 logits, CE on correct full byte ID remains unchanged.
Auxiliary loss is mean negative log probability of correct bit,
conditional on the 256 byte IDs (IDs9..264), eight bit marginals
computed by stable logsumexp. CE still penalizes special IDs.
No separate head, no new weights, no answer leakage, same inference
unconstrained argmax. No claim of carry reasoning from bit loss.
Two arms CE and CE+1*bitNLL from same 90-step learned old foundation,
same E fixed at anchor, same AdamW moments and RNG, same mixed
3840 targets/120 updates for all four ops, two seeds42/43.
Report exact and bit accuracy, CE, invalid IDs, majority by tier/family.
Additional loss compute means matched sample/step budget, not exact FLOPs.
No held-out test evaluation, no growth, no GPU. Local PyTorch objective
tests run independently of Transformers, full model CI required.
Run python -m vlanix_tiny.bit_learning.
