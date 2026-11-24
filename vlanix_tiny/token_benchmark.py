"""P0.7B: compare tokenizers without training a language model."""
import argparse
import hashlib
import json
import math
import unicodedata
from pathlib import Path
from tokenizers import Tokenizer, AddedToken, models, trainers, pre_tokenizers, decoders
from vlanix_tiny.tokenizer import SPECIAL
from vlanix_tiny.logic import generate, FAMILIES, verify

VI_PROBES = ("Tiếng Việt có dấu.", "Nếu A lớn hơn B thì B nhỏ hơn A.",
             unicodedata.normalize("NFD", "Tôi học toán."), "Số âm: −123,45.")
EN_PROBES = ("The sum of two even integers is even.",
             "If no evidence supports the claim, the answer is unknown.",
             "Calculate -123.45 + 6.78 = -116.67.", "A != B; x >= 10; (P AND Q).")
BUDGETS = (64,128,256,512)

def fit(texts, vocab, digits):
    if vocab not in (2048,4096): raise ValueError("unsupported vocab")
    texts=list(texts)
    if not texts or any(not isinstance(s,str) or not s.strip() or "<|" in s for s in texts):
        raise ValueError("invalid corpus")
    tok=Tokenizer(models.BPE(unk_token=SPECIAL[3]))
    byte=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),byte]) if digits else byte
    tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator(texts,trainers.BpeTrainer(vocab_size=vocab,min_frequency=2,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),special_tokens=list(SPECIAL),show_progress=False))
    missing=vocab-tok.get_vocab_size()
    if missing<0: raise ValueError("vocab overflow")
    if missing: tok.add_special_tokens([AddedToken(f"<|reserved_{i:04d}|>",special=True) for i in range(missing)])
    if tok.get_vocab_size()!=vocab or any(tok.token_to_id(t)!=i for i,t in enumerate(SPECIAL)):
        raise ValueError("token contract")
    return tok

def measure(tok,texts,budgets=BUDGETS):
    texts=list(texts)
    if not texts: raise ValueError("empty evaluation")
    lengths=[]; unknown=0; bad=0
    for s in texts:
        ids=tok.encode(s).ids
        lengths.append(len(ids)+2)
        unknown+=ids.count(3)
        bad+=tok.decode(ids)!=s
    ordered=sorted(lengths)
    return dict(rows=len(texts),tokens=sum(lengths),mean=round(sum(lengths)/len(texts),3),
        p95=ordered[math.ceil(.95*len(texts))-1],max=max(lengths),
        utf8_bytes=sum(len(s.encode()) for s in texts),
        roundtrip_failures=bad,unk_tokens=unknown,
        fit={str(b):sum(n<=b for n in lengths) for b in budgets})

def benchmark(out,per_family=256,seed=42):
    rows=generate(per_family,seed)
    if not all(verify(r) for r in rows): raise ValueError("bad logic fixtures")
    train=[r['text'] for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='validation']
    # Test rows remain unused for fit, metrics and selection.
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='ngc-token-benchmark-v1',seed=seed,train_rows=len(train),
        validation_rows=len(val),test_rows_used=0,freeze_ready=False,candidates=[])
    for name,vocab,digits in [('A',2048,False),('B',2048,True),('C',4096,False),('D',4096,True)]:
        tok=fit(train,vocab,digits)
        path=out/(name+'.json'); tok.save(str(path))
        m=dict(name=name,vocab=vocab,digit_isolation=digits,
            reserved=sum(t.startswith('<|reserved_') for t in tok.get_vocab()),
            embedding_params=vocab*256,total_params=5017600+vocab*256,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            validation=measure(tok,[r['text'] for r in val]),
            by_family={f:measure(tok,[r['text'] for r in val if r['family']==f]) for f in FAMILIES},
            english_probe=measure(tok,EN_PROBES),vietnamese_probe=measure(tok,VI_PROBES),
            digit_probe=[len(tok.encode(s).ids) for s in ('1234567890','-123.45','20261125')])
        if any(x['roundtrip_failures'] or x['unk_tokens'] for x in (m['validation'],m['english_probe'],m['vietnamese_probe'])):
            raise ValueError("lossless encoding failed")
        report['candidates'].append(m)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--per-family',type=int,default=256); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    print(json.dumps(benchmark(a.output,a.per_family,a.seed),sort_keys=True))
if __name__=='__main__': main()
