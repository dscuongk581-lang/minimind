# VLANIX Tiny P0.1

Research-only Dense MiniMind derivative. No changes to upstream model or trainer.
Random initialization, no pretrained checkpoint and no production integration.

| Setting | Value |
|---|---:|
| Hidden size | 256 |
| Layers | 6 |
| Query / KV heads | 4 / 2 |
| FFN | 832 |
| Vocabulary budget | 4096 |
| Context | 512 |
| Expected unique parameters | 6066176 |

Parameter calculation: 4096*256 + 6*(3*256^2 + 3*256*832 + 2*256 + 2*64) + 256.
FP16 weights: 12,132,352 bytes, excluding runtime and cache.

## CPU acceptance
From repo root, in a Python 3.11 environment:
```sh
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.6
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p test_vlanix_tiny.py -v
```

P0.1 acceptance requires actual parameter count, finite forward/backward,
causal prefix invariance and cache equivalence. Tests are not evidence until run.
The 4096 vocabulary is a budget, not a trained tokenizer. Do not pair this
model with the original 6400-token tokenizer or checkpoints.

P0.2: train a separate Vietnamese byte-level BPE tokenizer with explicit
special IDs, measure token efficiency and round-trip fidelity on held-out text.
P0.3: licensed Vietnamese corpus, deduplication, document-level train/validation
split and reproducible training. Do not commit private data or weights.
