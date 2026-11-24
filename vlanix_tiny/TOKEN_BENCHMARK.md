# P0.7B — English tokenizer diagnostics
Run python -m vlanix_tiny.token_benchmark --output runs/token-benchmark-v1.
Four byte-level BPE variants: A/B = 2048, C/D = 4096, B/D isolate digits.
All preserve special IDs and byte alphabet; no normalization inside tokenizer
(probes verify exact NFC and NFD strings). Fit ONLY English train rows; evaluate
validation separately. Official test rows unused. Vietnamese probes are never
used for fitting; future adaptation at 30M is optional. Length metrics include
BOS/EOS and exclude padding. Fit counts measure atomic problem coverage;
no chunk truncation. SHA identifies each saved tokenizer, not a guarantee of
reproducible byte-for-byte retraining across platforms.

These are small procedural fixtures with repetitive templates, not sufficient
vocabulary coverage. Reserved tokens fill unused capacity but must not be
generated. No candidate is frozen. Next: diverse reviewed English corpus,
independent validation, answer-accuracy CPU pilots at controlled budgets,
chunk packing and growth gates. Keep original Tiny model and tokenizer APIs
unchanged. 2048 is exploratory; smaller embeddings mean fewer total parameters
unless the architecture is changed. Compare both equal backbone and controlled
parameter/compute budgets. No GPU or model training in this benchmark.
