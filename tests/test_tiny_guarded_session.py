import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from vlanix_tiny.binary_logic import generate
from vlanix_tiny.binary_state import State
from vlanix_tiny.guarded_session import Session
from vlanix_tiny.binary_homeostasis import answer_step

class GuardTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        rows=[r for r in generate(2) if r['split']=='train']
        s=State(); answer_step(s,rows[:2]); s.grow(2,16)
        return s,rows[2:4],[rows[4:6],rows[6:8]]

    def test_resume_and_plan_binding(self):
        s,cal,plan=self.fixture()
        a=Session(s,cal,plan,'fixture',cap=10.)
        self.assertTrue(a.attempt()['accepted'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.pt'; a.save(p)
            with self.assertRaises(ValueError): Session.restore(p,cal,plan[::-1],'fixture')
            b=Session.restore(p,cal,plan,'fixture')
            self.assertEqual(a.pi.snapshot(),b.pi.snapshot())
            self.assertEqual(b.cursor,1)
            self.assertTrue(torch.equal(a.ref,b.ref))
            self.assertEqual(a.attempt(),b.attempt())
            for x,y in zip(a.state.model.parameters(),b.state.model.parameters()):
                self.assertTrue(torch.equal(x,y))

    def test_rejected_step_rolls_back(self):
        s,cal,plan=self.fixture()
        a=Session(s,cal,plan,'fixture',cap=.075)
        weights=[p.detach().clone() for p in a.state.model.parameters()]
        pi=a.pi.snapshot(); rng=torch.get_rng_state(); steps=s.steps
        moments=[(v['step'].clone(),v['exp_avg'].clone(),v['exp_avg_sq'].clone()) for v in s.opt.state.values()]
        with patch('vlanix_tiny.guarded_session.drift',side_effect=[.04,.2]):
            result=a.attempt()
        self.assertFalse(result['accepted'])
        self.assertEqual(a.cursor,0)
        self.assertTrue(a.blocked)
        self.assertEqual(a.state.steps,steps)
        self.assertEqual(a.pi.snapshot(),pi)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for p,q in zip(weights,a.state.model.parameters()): self.assertTrue(torch.equal(p,q))
        self.assertEqual(len(moments),len(a.state.opt.state))
        for expected,v in zip(moments,a.state.opt.state.values()):
            for x,key in zip(expected,('step','exp_avg','exp_avg_sq')):
                self.assertTrue(torch.equal(x,v[key]))
        with self.assertRaises(RuntimeError): a.attempt()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.pt'; a.save(p)
            self.assertTrue(Session.restore(p,cal,plan,'fixture').blocked)

    def test_disjoint_plan(self):
        s,cal,plan=self.fixture()
        with self.assertRaises(ValueError): Session(s,cal,[cal],'fixture')
