import unittest
import torch
from vlanix_tiny.config import make_model
from vlanix_tiny.organic import add_microblock, observe, count_params, run_probe

class OrganicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=make_model()
        self.x=torch.tensor([[1,11,12,13,2],[1,14,15,16,2]])
    def test_conservation_and_learning(self):
        m=self.model
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        m(self.x,labels=self.x).loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        old=dict(m.named_parameters())
        states={p:opt.state[p]['exp_avg'].clone() for p in old.values()}
        y=m(self.x,labels=self.x)
        y.loss.backward()
        grads={n:p.grad.clone() for n,p in old.items()}
        logits=y.logits.detach().clone()
        opt.zero_grad(set_to_none=True)
        block=add_microblock(m,2,32,opt)
        self.assertEqual(count_params(m),6066176+24576)
        self.assertIs(m.model.embed_tokens.weight,m.lm_head.weight)
        for n,p in old.items():
            self.assertIs(dict(m.named_parameters())[n],p)
            self.assertTrue(torch.equal(opt.state[p]['exp_avg'],states[p]))
        new=m(self.x,labels=self.x)
        torch.testing.assert_close(new.logits,logits,rtol=0,atol=0)
        new.loss.backward()
        for n,p in old.items():
            torch.testing.assert_close(p.grad,grads[n],rtol=1e-5,atol=1e-7)
        self.assertGreater(block.down_proj.weight.grad.norm().item(),0)
        self.assertEqual(block.up_proj.weight.grad.norm().item(),0)
        opt.step(); opt.zero_grad(set_to_none=True)
        m(self.x,labels=self.x).loss.backward()
        self.assertGreater(block.up_proj.weight.grad.norm().item(),0)
    def test_sensor_is_read_only(self):
        m=self.model
        out=observe(m,{'a':[self.x]*4,'b':[self.x.flip(0)]*4})
        self.assertEqual(len(out),6)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        self.assertTrue(all(abs(x['moments']['closure_error'])<1e-10 for x in out))
    def test_invalid_growth(self):
        opt=torch.optim.AdamW(self.model.parameters())
        for layer,width in ((-1,32),(6,32),(0,0),(0,40000)):
            with self.assertRaises(ValueError): add_microblock(self.model,layer,width,opt)
        self.assertEqual(count_params(self.model),6066176)
    def test_probe(self):
        r=run_probe()
        self.assertEqual(r['params_after']-r['params_before'],24576)
        self.assertEqual(r['logit_max_delta'],0)
        self.assertLess(r['gradient_max_delta'],1e-6)
        self.assertGreater(r['new_down_grad_norm'],0)
        self.assertFalse(r['growth_authorized'])
