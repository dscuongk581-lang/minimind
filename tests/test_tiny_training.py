import tempfile
import unittest
from pathlib import Path
import torch
from vlanix_tiny.training import Trainer, encode_documents
from vlanix_tiny.tokenizer import train_tokenizer

class TrainingTests(unittest.TestCase):
    def test_padding_and_eos(self):
        with tempfile.TemporaryDirectory() as d:
            tok=train_tokenizer(["Tiếng Việt rất đẹp."], Path(d)/"tok.json")
            x=encode_documents(tok, ["Tiếng Việt rất đẹp."], 32)
            self.assertEqual(x[0,0].item(),1)
            self.assertIn(2,x[0].tolist())
            self.assertEqual(x[0,-1].item(),0)

    def test_loss_decreases_and_resume(self):
        x=torch.tensor([[1,20,21,22,20,21,22,2,0,0]]*2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"state.pt"
            a=Trainer("cpu",42,0.001)
            initial=a.evaluate(x,2)
            for _ in range(5):
                a.step(x)
            self.assertLess(a.evaluate(x,2),initial)
            a.save(p,"test")
            b=Trainer("cpu",42,0.001)
            b.restore(p,"test")
            self.assertEqual(b.steps,5)
            self.assertAlmostEqual(a.evaluate(x,2),b.evaluate(x,2),places=6)
            a.step(x)
            b.step(x)
            for p1,p2 in zip(a.model.parameters(),b.model.parameters()):
                torch.testing.assert_close(p1,p2)
            with self.assertRaises(ValueError):
                b.restore(p,"different")
