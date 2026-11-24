import unittest
import torch
from vlanix_tiny.config import make_model, EXPECTED_PARAMS

class TinyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = make_model()

    def test_parameter_budget(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), EXPECTED_PARAMS)
        self.assertIs(self.model.model.embed_tokens.weight, self.model.lm_head.weight)

    def test_forward_backward(self):
        x = torch.randint(3, 4096, (2, 16))
        out = self.model(x, labels=x)
        self.assertEqual(tuple(out.logits.shape), (2, 16, 4096))
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_cache_matches_full(self):
        self.model.eval()
        x = torch.randint(3, 4096, (1, 12))
        with torch.no_grad():
            full = self.model(x).logits[:, -1]
            first = self.model(x[:, :-1], use_cache=True)
            cached = self.model(x[:, -1:], past_key_values=first.past_key_values, use_cache=True)
        torch.testing.assert_close(full, cached.logits[:, -1], atol=1e-4, rtol=1e-4)

    def test_causal_prefix(self):
        self.model.eval()
        x = torch.randint(3, 4096, (1, 12))
        y = x.clone()
        y[:, 6:] = (y[:, 6:] + 1) % 4096
        with torch.no_grad():
            torch.testing.assert_close(self.model(x).logits[:, :6], self.model(y).logits[:, :6])

if __name__ == "__main__":
    unittest.main()
