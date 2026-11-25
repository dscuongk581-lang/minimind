# P0.11B — tied embedding conservation
P0.11A CI 45563317058 passed 100 tests. Bit curriculum hard AND
39/192 vs raw curriculum 22/192, seed variance substantial; OOD AND/XOR
zero. Reproduce a bit-initialized foundation for seeds42/43 with 90
curriculum updates. In-memory checkpoint includes all model weights,
AdamW moments and CPU RNG. Both optimizer groups use LR .0005 in warmup.
Each continuation starts from the same checkpoint and receives the
same mixed train order (90 updates; 2880 answer targets).
Only embedding group LR varies: 0, .00005, .0005. E stays tied across
input/output. Zero LR keeps weights fixed exactly while gradient and
AdamW moment computation continue; not a compute-saving freeze.
Global clipping includes E in all arms. Report exact E equality, relative
L2 displacement, maximum coordinate displacement, sign agreement of the
first eight byte dimensions, anchor and end E hashes, order digest,
per-family/tier validation CE, exact and bit accuracy, invalid IDs,
and KL from the foundation predictions. KL is descriptive, no guard.
No test use, no new task transfer (continues the same train corpus),
no GPU, production or growth. Two seeds and repeated validation are
exploratory, not confirmatory; no winner declaration without new evidence.
Run python -m vlanix_tiny.embedding_conservation.
