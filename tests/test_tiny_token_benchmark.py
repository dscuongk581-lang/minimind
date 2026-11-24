import tempfile
import unittest
from pathlib import Path
from vlanix_tiny.token_benchmark import fit, measure, benchmark, VI_PROBES
from vlanix_tiny.logic import generate

class TokenBenchmarkTests(unittest.TestCase):
    def test_digits_and_utf8(self):
        rows=generate(32,42)
        tok=fit([r['text'] for r in rows if r['split']=='train'],2048,True)
        self.assertEqual(tok.get_vocab_size(),2048)
        self.assertEqual(len(tok.encode('1234567890').ids),10)
        for s in VI_PROBES:
            self.assertEqual(tok.decode(tok.encode(s).ids),s)
    def test_measure_whole_problem_budget(self):
        tok=fit(['abc 123']*3,2048,True)
        m=measure(tok,['abc 123','abc 123'],budgets=(2,64))
        self.assertEqual(m['fit']['2'],0)
        self.assertEqual(m['fit']['64'],2)
        self.assertEqual(m['roundtrip_failures'],0)
        self.assertEqual(m['unk_tokens'],0)
    def test_report_and_no_test_use(self):
        with tempfile.TemporaryDirectory() as d:
            report=benchmark(Path(d)/'bench',per_family=16)
            self.assertEqual(report['train_rows'],64)
            self.assertEqual(report['validation_rows'],64)
            self.assertEqual(report['test_rows_used'],0)
            self.assertEqual(len(report['candidates']),4)
            self.assertFalse(report['freeze_ready'])
            self.assertTrue(all(x['validation']['roundtrip_failures']==0 for x in report['candidates']))
    def test_invalid_fit(self):
        for texts in ([],[''],['<|pad|>'],['<|reserved_0000|>']):
            with self.assertRaises(ValueError): fit(texts,2048,False)
