import tempfile
import unittest
from vlanix_tiny.dataset_benchmark import sample_shards, benchmark, density

class BenchmarkTests(unittest.TestCase):
    def test_balanced_shards_and_long_skip(self):
        streams=[[dict(id=str(i*100+j),url="https://vi.wikipedia.org/wiki/Test",text=("x"*13000 if j==0 else f"Bài học tiếng Việt {i} số {j}.")) for j in range(12)] for i in range(4)]
        rows,stats=sample_shards(streams,12,20000)
        self.assertEqual(len(rows),44)
        self.assertEqual([s["accepted"] for s in stats],[11]*4)
        self.assertEqual([s["length_rejected"] for s in stats],[1]*4)
    def test_density(self):
        self.assertEqual(density([3]*8,16),dict(tokens=24,unpacked_rows=8,packed_rows=2,unpacked_density=0.1875,packed_density=0.75))
    def test_all_budgets_same_documents(self):
        streams=[[dict(id=str(i*100+j),url="https://vi.wikipedia.org/wiki/Test",text=f"Bài học tiếng Việt thứ {i} số {j}. "+"Máy tính xử lý dữ liệu. "*(1+j%3)) for j in range(20)] for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            report=benchmark(streams,d,20,500000)
        self.assertEqual(set(report["budgets"]),{"64","128","256","512"})
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["gate"]["ready"])
        self.assertTrue(all(v["train"]["tokens"]>0 for v in report["budgets"].values()))
