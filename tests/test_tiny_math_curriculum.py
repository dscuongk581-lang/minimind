import unittest
from collections import Counter
import torch
from vlanix_tiny.math_curriculum import generate,audit,schedule,solve,oracle
from vlanix_tiny.math_learning import bit_initialize,tensorize
from vlanix_tiny.binary_model import make_binary_model

class MathCurriculumTests(unittest.TestCase):
    def test_exhaustive_oracle(self):
        for op in (1,2,7):
            for a in range(256):
                for b in range(256):
                    self.assertEqual(solve(op,a,b),oracle(op,a,b))
    def test_split(self):
        rows=generate(42); report=audit(rows)
        self.assertEqual(rows,generate(42))
        self.assertEqual(report['train_byte_coverage'],256)
        self.assertEqual(report['counts'],{
            'train/easy':288, 'validation/easy':72, 'test/easy':48,
            'train/hard':1152, 'validation/hard':288, 'test/hard':288,
            'validation/ood':288, 'test/ood':288})
        self.assertEqual(sum(report['counts'].values()),len(rows))
        self.assertEqual(len(rows),3*(136+384+96+96+96+96))
        pair={}
        for row in rows:
            q=bytes.fromhex(row['query'])
            key=tuple(sorted(q[1:]))
            self.assertEqual(pair.setdefault(key,row['split']),row['split'])
    def test_schedule(self):
        rows=generate()
        a=schedule(rows,'mixed',4); b=schedule(rows,'curriculum',4)
        self.assertEqual(Counter(r['id'] for r in a),Counter(r['id'] for r in b))
        self.assertEqual(len(a),2880)
        self.assertTrue(all(r['split']=='train' for r in b))
        self.assertTrue(all(r['tier']=='easy' for r in b[:576]))
        self.assertTrue(all(r['tier']=='hard' for r in b[576:]))
    def test_bits(self):
        torch.manual_seed(42); m=make_binary_model()
        w=m.lm_head.weight.detach().clone()
        bit_initialize(m)
        e=m.model.embed_tokens.weight.detach()
        self.assertTrue(torch.equal(e[:9],w[:9]))
        torch.testing.assert_close(e.norm(dim=1),w.norm(dim=1),atol=1e-6,rtol=1e-5)
        self.assertTrue((e[9,:8]<0).all())
        self.assertTrue((e[264,:8]>0).all())
        self.assertTrue((e[10,0]>0) and (e[10,1:8]<0).all())
        self.assertIs(m.model.embed_tokens.weight,m.lm_head.weight)
    def test_prompt(self):
        r=generate()[0]; x,y=tensorize([r])
        self.assertEqual(x.tolist(),[[1]+[b+9 for b in bytes.fromhex(r['query'])]])
        self.assertEqual(y.tolist(),[r['answer']+9])
