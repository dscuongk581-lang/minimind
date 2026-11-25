import copy
import unittest
import torch
from vlanix_tiny.adaptive_guard import backtrack

class AdaptiveGuardTests(unittest.TestCase):
    def fixture(self):
        m=torch.nn.Linear(1,1,bias=False); m.weight.data.fill_(1.)
        opt=torch.optim.AdamW(m.parameters(),lr=.1,weight_decay=0.)
        def update():
            opt.zero_grad(set_to_none=True); m.weight.sum().backward(); opt.step()
            torch.rand(2); return 1.
        return m,opt,update
    def test_retry(self):
        m,opt,update=self.fixture()
        r=backtrack(m,opt,update,lambda: m.weight.item()>.95)
        self.assertEqual(r['attempts'],2); self.assertEqual(r['scale'],.25)
        self.assertAlmostEqual(m.weight.item(),.975,places=6)
        self.assertEqual(opt.param_groups[0]['lr'],.1)
        self.assertEqual(opt.state[m.weight]['step'].item(),1.)
    def test_reject_restores(self):
        m,opt,update=self.fixture(); update()
        before=m.weight.detach().clone(); state=copy.deepcopy(opt.state_dict()); rng=torch.get_rng_state().clone()
        r=backtrack(m,opt,update,lambda: False)
        self.assertIsNone(r['scale']); self.assertEqual(r['attempts'],4)
        self.assertTrue(torch.equal(m.weight,before)); self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for k,v in state['state'][0].items(): self.assertTrue(torch.equal(v,opt.state_dict()['state'][0][k]))
        self.assertEqual(opt.param_groups[0]['lr'],.1)
    def test_full_accept(self):
        m,opt,update=self.fixture(); r=backtrack(m,opt,update,lambda: True)
        self.assertEqual(r['scale'],1.); self.assertEqual(r['attempts'],1)
        self.assertAlmostEqual(m.weight.item(),.9,places=6)
    def test_exception_and_validation(self):
        m,opt,update=self.fixture(); before=m.weight.detach().clone()
        def fail():
            update(); raise RuntimeError('failure')
        with self.assertRaises(RuntimeError): backtrack(m,opt,fail,lambda:True)
        self.assertTrue(torch.equal(before,m.weight)); self.assertEqual(opt.param_groups[0]['lr'],.1)
        with self.assertRaises(ValueError): backtrack(m,opt,update,lambda:True,(1.,0.))
        with self.assertRaises(ValueError): backtrack(m,opt,update,lambda:True,(.25,1.))
