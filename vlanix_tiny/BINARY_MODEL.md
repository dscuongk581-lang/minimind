# P0.9D — binary neural core
Independent candidate; no changes to BPE or upstream configuration defaults.
Fixed 265 IDs, tied E[265,256], 6 dense layers, FFN width 1024, GQA 4/2,
5,970,176 unique trainable parameters. The FFN width reallocates most of
the embedding savings; this is a design choice, not a fair performance
comparison to BPE yet. Embedding starts random and is learned, not frozen.
Right-pad with 0; labels ignore all prompt/pad positions, supervise one
answer byte and EOS. Causal attention makes right padding invisible to
supervised positions. Answer accuracy is unconstrained argmax on query-only
input; control IDs count as wrong. Loss is teacher-forced answer+EOS CE;
EOS learning alone may reduce loss without useful answer prediction.
One shuffled epoch on 768 train rows, batch16, AdamW .001, clip1,
seed42, compare before/after on 768 OOD validation rows, by family.
Train-family majority comparator; no test rows used for model metrics.
Data generation includes test for integrity; not used to tune or train.
No saved trained model, no GPU, no autonomous growth, no Rust training.
P0.9E: matched fixed/growth trials and topology/optimizer restore;
compare representation at matched budgets and multiple seeds.
Run python -m vlanix_tiny.binary_model.
