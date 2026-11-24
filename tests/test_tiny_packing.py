import unittest
import torch
from vlanix_tiny.config import make_model
from vlanix_tiny.packing import pack, packed_loss

class PackingTests(unittest.TestCase):
    def test_density_and_no_loss_of_segments(self):
        seqs=[[1,10+i,2] for i in range(8)]
        batch=pack(seqs,16)
        self.assertEqual(batch.ids.shape,(2,16))
        self.assertEqual(batch.target_count,16)
        self.assertEqual(batch.used_tokens,24)
        self.assertEqual(batch.ids[batch.segments>=0].tolist(),sum(seqs,[]))

    def test_loss_matches_independent_sequences(self):
        torch.manual_seed(42)
        model=make_model().eval()
        seqs=[[1,10,11,2],[1,20,21,22,2]]
        b=pack(seqs,16)
        with torch.no_grad():
            loss=packed_loss(model,b)
            expected=sum(model(torch.tensor([s]),labels=torch.tensor([s])).loss*(len(s)-1) for s in seqs)/sum(len(s)-1 for s in seqs)
            torch.testing.assert_close(loss,expected,atol=1e-5,rtol=1e-5)

    def test_gradient_matches_independent_sequences(self):
        torch.manual_seed(42)
        model=make_model()
        seqs=[[1,10,11,2],[1,20,21,2]]
        packed_loss(model,pack(seqs,16)).backward()
        grads=[p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        sum(model(torch.tensor([s]),labels=torch.tensor([s])).loss/2 for s in seqs).backward()
        for p,g in zip(model.parameters(),grads):
            torch.testing.assert_close(p.grad,g,atol=2e-5,rtol=1e-4)

    def test_invalid_sequences(self):
        for seq in ([1,0,2],[1,1,2],[1,9,2,2],[1,5000,2],[1,2], [9,10,2]):
            with self.assertRaises(ValueError):
                pack([seq],16)
