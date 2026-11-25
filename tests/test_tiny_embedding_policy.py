import unittest
import torch
from torch import nn
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(265,256)
        self.core=nn.Linear(256,256,bias=False)
        self.lm_head=nn.Linear(256,265,bias=False)
        self.lm_head.weight=self.model.embed_tokens.weight
    def forward(self,x):
        return self.lm_head(self.core(self.model.embed_tokens(x)))

class EmbeddingPolicyTests(unittest.TestCase):
    def test_frozen_step(self):
        torch.manual_seed(42); m=Toy(); o=make_optimizer(m,.001)
        e=m.lm_head.weight; before=e.detach().clone(); core=m.core.weight.detach().clone()
        set_policy(o,0.)
        m(torch.tensor([10,20])).square().mean().backward(); o.step()
        self.assertTrue(torch.equal(e,before))
        self.assertFalse(torch.equal(m.core.weight,core))
        self.assertIs(m.model.embed_tokens.weight,e)
        self.assertEqual(geometry(before,e)['relative_l2'],0.)
    def test_slow_and_full(self):
        torch.manual_seed(43); m=Toy(); o=make_optimizer(m,.001)
        set_policy(o,.1)
        self.assertAlmostEqual(o.param_groups[1]['lr'],.0001)
        self.assertEqual(o.param_groups[0]['lr'],.001)
        self.assertEqual(sum(len(g['params']) for g in o.param_groups),len(list(m.parameters())))
        before=m.lm_head.weight.detach().clone()
        m(torch.tensor([10,20])).square().mean().backward(); o.step()
        self.assertGreater(geometry(before,m.lm_head.weight)['relative_l2'],0.)
        set_policy(o,1.)
        self.assertEqual(o.param_groups[1]['lr'],.001)
        with self.assertRaises(ValueError): set_policy(o,-1.)
    def test_untied_rejected(self):
        m=Toy(); m.lm_head.weight=nn.Parameter(m.lm_head.weight.detach().clone())
        with self.assertRaises(ValueError): make_optimizer(m,.001)
