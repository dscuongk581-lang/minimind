import copy
import unittest
import torch
from vlanix_tiny.learning_state import transition_optimizer

class LearningStateTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(7)
        model=torch.nn.Linear(2,1)
        opt=torch.optim.AdamW([{'params':[model.weight],'role':'core'},{'params':[model.bias],'role':'embedding','lr':0.}],lr=.01)
        for _ in range(3):
            opt.zero_grad(set_to_none=True); model(torch.ones(1,2)).sum().backward(); opt.step()
        opt.zero_grad(set_to_none=True)
        return model,opt,copy.deepcopy(opt.state_dict())
    def test_preserve(self):
        m,opt,state=self.fixture()
        before=[p.clone() for p in m.parameters()]
        info=transition_optimizer(opt,'retain')
        self.assertEqual(info['after_slots'],2)
        for p,b in zip(m.parameters(),before): self.assertTrue(torch.equal(p,b))
        for k,v in state['state'].items():
            for n,t in v.items(): self.assertTrue(torch.equal(opt.state_dict()['state'][k][n],t))
    def test_reset(self):
        m,opt,state=self.fixture(); before=[p.clone() for p in m.parameters()]; rng=torch.get_rng_state().clone()
        info=transition_optimizer(opt,'reset')
        self.assertEqual(info,{'before_slots':2,'after_slots':0})
        self.assertEqual(opt.state_dict()['param_groups'],state['param_groups'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for p,b in zip(m.parameters(),before): self.assertTrue(torch.equal(p,b))
        opt.zero_grad(); m(torch.ones(1,2)).sum().backward(); opt.step()
        self.assertEqual(opt.state[m.weight]['step'].item(),1.)
        self.assertTrue(torch.equal(m.bias,before[1]))
    def test_invalid(self):
        m,opt,state=self.fixture()
        with self.assertRaises(ValueError): transition_optimizer(opt,'other')
        self.assertEqual(len(opt.state),2)
    def test_empty(self):
        opt=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
        self.assertEqual(transition_optimizer(opt,'reset'),{'before_slots':0,'after_slots':0})
