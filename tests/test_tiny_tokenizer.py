import tempfile
import unittest
from pathlib import Path
from vlanix_tiny.tokenizer import train_tokenizer, load_tokenizer, SPECIAL

class TokenizerTests(unittest.TestCase):
    def test_roundtrip_and_contract(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tokenizer.json"
            tok = train_tokenizer(["Tiếng Việt có dấu. CPU và RAM.", "Hà Nội\t🙂\n"], path)
            tok = load_tokenizer(path)
            self.assertEqual(tok.get_vocab_size(), 4096)
            for i, token in enumerate(SPECIAL):
                self.assertEqual(tok.token_to_id(token), i)
            for s in ["  Tiếng Việt\n", "a\u0301", "🙂\tCPU", "未知文字"]:
                ids = tok.encode(s).ids
                self.assertNotIn(3, ids)
                self.assertEqual(tok.decode(ids, skip_special_tokens=False), s)

    def test_reject_empty_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                train_tokenizer(["", "  "], Path(d)/"tokenizer.json")

    def test_reserved_literals_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                train_tokenizer(["văn bản <|pad|>"], Path(d)/"tokenizer.json")
