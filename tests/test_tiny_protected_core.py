import copy
import unittest
import torch
from vlanix_tiny.protected_core import guarded_update, pack_prompt

class ProtectedCoreTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(7)
        m=torch.nn.Linear(1,1,bias=False); m.weight.data.fill_(1.)
        opt=torch.optim.AdamW(m.parameters(),lr=.1)
        return m,opt
    def test_accept(self):
        m,opt=self.fixture()
        def update():
            opt.zero_grad(); (-m.weight.sum()).backward(); opt.step(); return 2.
        accepted,loss=guarded_update(m,opt,update,lambda: bool(m.weight.item()>0))
        self.assertTrue(accepted); self.assertEqual(loss,2.); self.assertGreater(m.weight.item(),1.)
    def test_rollback(self):
        m,opt=self.fixture()
        opt.zero_grad(); (-m.weight.sum()).backward(); opt.step(); opt.zero_grad(set_to_none=True)
        before=m.weight.detach().clone(); state=copy.deepcopy(opt.state_dict()); rng=torch.get_rng_state().clone()
        def update():
            opt.zero_grad(); m.weight.sum().backward(); opt.step(); torch.rand(3); m.weight.data.fill_(-1.); return 3.
        accepted,_=guarded_update(m,opt,update,lambda: bool(m.weight.item()>0))
        self.assertFalse(accepted); self.assertTrue(torch.equal(m.weight,before)); self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for k,v in state['state'][0].items(): self.assertTrue(torch.equal(v,opt.state_dict()['state'][0][k]))
        self.assertIsNone(m.weight.grad)
    def test_exception(self):
        m,opt=self.fixture(); before=m.weight.detach().clone()
        def update():
            m.weight.data.zero_(); raise RuntimeError('bad')
        with self.assertRaises(RuntimeError): guarded_update(m,opt,update,lambda: True)
        self.assertTrue(torch.equal(m.weight,before))
    def test_pack(self):
        self.assertEqual(pack_prompt([9,10,11,12,9,10,11,12]),[1,18,9,10,11,12,9,10,11,12])
        self.assertIsNone(pack_prompt([9]*7+[13])); self.assertIsNone(pack_prompt([9]*7))
