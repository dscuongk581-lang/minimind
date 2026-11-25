import unittest
import torch
from vlanix_tiny.binary_logic import generate
from vlanix_tiny.binary_model import make_binary_model,batch,prompt_batch,predict,evaluate,train_step,EXPECTED_PARAMS

class BinaryModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.rows=generate(1,42)
    def test_shape_and_tying(self):
        m=make_binary_model()
        self.assertEqual(sum(p.numel() for p in m.parameters()),EXPECTED_PARAMS)
        self.assertEqual(tuple(m.model.embed_tokens.weight.shape),(265,256))
        self.assertIs(m.model.embed_tokens.weight,m.lm_head.weight)
    def test_prompt_and_mask(self):
        rows=[r for r in self.rows if r['split']=='train']
        x,y=batch(rows); p,lengths=prompt_batch(rows)
        for i,r in enumerate(rows):
            n=int(lengths[i])
            self.assertEqual(p[i,:n].tolist(),[1]+[b+9 for b in bytes.fromhex(r['query'])])
            self.assertTrue(torch.equal(p[i,:n],x[i,:n]))
            self.assertTrue((y[i,:n]==-100).all())
            self.assertEqual(y[i,n:n+2].tolist(),[r['answer']+9,2])
            self.assertTrue((y[i,n+2:]==-100).all())
        self.assertEqual(int((y[:,1:]!=-100).sum()),2*len(rows))
    def test_forward_backward_and_holdout(self):
        m=make_binary_model()
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        train=[r for r in self.rows if r['split']=='train']
        val=[r for r in self.rows if r['split']=='validation']
        w=m.lm_head.weight.detach().clone()
        result=train_step(m,opt,train)
        self.assertEqual(result['targets'],12)
        self.assertGreater(result['grad_norm'],0)
        self.assertFalse(torch.equal(w,m.lm_head.weight))
        with self.assertRaises(ValueError): train_step(m,opt,val)
        self.assertEqual(evaluate(m,val)['rows'],6)
        self.assertTrue(m.training)
    def test_causal_prompt_parity(self):
        m=make_binary_model().eval()
        rows=[r for r in self.rows if r['split']=='validation']
        x,y=batch(rows); p,lengths=prompt_batch(rows)
        with torch.no_grad():
            a=m(x).logits[torch.arange(len(rows)),lengths-1]
            b=m(p).logits[torch.arange(len(rows)),lengths-1]
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            self.assertTrue(torch.equal(predict(m,rows),b.argmax(-1)))
