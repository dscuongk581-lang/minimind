# P0.2 — Vietnamese tokenizer

Run from repo root:
```sh
pip install transformers==4.57.6
python -m vlanix_tiny.tokenizer --corpus vlanix_tiny/data/tokenizer_smoke.jsonl --output /tmp/tiny-tokenizer.json
python -m unittest discover -s tests -p test_tiny_tokenizer.py -v
```
The seven original Vietnamese sentences are a smoke fixture, NOT a training corpus.
A 4096-ID budget includes nine special tokens, byte alphabet, learned merges
and temporary reserved placeholders. No claim of Vietnamese compression quality.
Reserved placeholders must be masked during generation; do not train a final
model on a smoke tokenizer. Final tokenizer must be trained on a licensed
representative corpus and frozen before pretraining. Do not use the upstream
6400-token tokenizer. No Unicode normalization: preserve exact input bytes.
Evaluate on held-out Vietnamese, English technical text and code.
