import tempfile
import unittest
from pathlib import Path
import torch
from vlanix_tiny.binary_state import State
from vlanix_tiny.binary_logic import generate
from vlanix_tiny.binary_model import batch
from vlanix_tiny.organic import count_params,observe

class StateTests(unittest.TestCase):
    def test_topology_resume(self):
        torch.manual_seed(42)
        rows=[r for r in generate(1) if r['split']=='train']
        s=State(); s.step(rows); s.grow(4,16); s.grow(0,32); s.step(rows)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt'; s.save(p,'fixture')
            with self.assertRaises(ValueError): State.restore(p,'other')
            t=State.restore(p,'fixture')
            self.assertEqual(t.topology,[[4,16],[0,32]])
            self.assertEqual(t.steps,2)
            self.assertEqual(count_params(t.model),5970176+768*48)
            self.assertEqual(len(t.opt.param_groups),3)
            x,_=batch(rows)
            with torch.no_grad():
                torch.testing.assert_close(t.model(x).logits,s.model(x).logits,rtol=0,atol=0)
            self.assertEqual(s.step(rows),t.step(rows))
            for a,b in zip(s.model.parameters(),t.model.parameters()):
                self.assertTrue(torch.equal(a,b))
    def test_budget_and_identity(self):
        s=State()
        with self.assertRaises(ValueError): s.grow(0,40000)
        self.assertEqual(s.topology,[])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.pt'; s.save(p,'fixture')
            a=torch.load(p,weights_only=True)
            a['params']+=1; torch.save(a,p)
            with self.assertRaises(ValueError): State.restore(p,'fixture')
    def test_sensor_masked(self):
        s=State(); rows=[r for r in generate(4) if r['split']=='train' and r['family']=='arithmetic']
        groups={'math':[batch([r]) for r in rows]}
        report=observe(s.model,groups)
        self.assertEqual(len(report),6)
        self.assertTrue(all(p.grad is None for p in s.model.parameters()))
