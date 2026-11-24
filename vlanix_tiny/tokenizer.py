"""Offline Vietnamese byte-level BPE research tokenizer."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer, AddedToken, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 4096
SPECIAL = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|system|>",
           "<|user|>", "<|assistant|>", "<|tool_call|>", "<|tool_response|>")

def train_tokenizer(texts, output):
    tok = Tokenizer(models.BPE(unk_token=SPECIAL[3]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    seen = False
    def checked():
        nonlocal seen
        for s in texts:
            if not isinstance(s, str):
                raise ValueError("expected text string")
            if any(t in s for t in SPECIAL) or "<|reserved_" in s:
                raise ValueError("reserved marker in corpus")
            if s.strip():
                seen = True
                yield s
    tok.train_from_iterator(checked(), trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        special_tokens=list(SPECIAL), show_progress=False))
    if not seen:
        raise ValueError("empty corpus")
    # Keep model shape fixed even for a tiny smoke corpus; placeholders
    # are not learned vocabulary and must be masked at generation time.
    missing = VOCAB_SIZE - tok.get_vocab_size()
    if missing:
        tok.add_special_tokens([AddedToken(f"<|reserved_{i:04d}|>", special=True)
                                for i in range(missing)])
    assert tok.get_vocab_size() == VOCAB_SIZE
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output))
    return tok

def load_tokenizer(path):
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size() != VOCAB_SIZE or any(tok.token_to_id(t) != i for i,t in enumerate(SPECIAL)):
        raise ValueError("incompatible Tiny tokenizer")
    return tok

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    tok = train_tokenizer(read_jsonl(args.corpus), args.output)
    print(json.dumps({"vocab_size": tok.get_vocab_size(),
        "learned_and_byte_tokens": sum(not t.startswith("<|reserved_") and t not in SPECIAL for t in tok.get_vocab()),
        "reserved": sum(t.startswith("<|reserved_") for t in tok.get_vocab())}))

if __name__ == "__main__":
    main()
