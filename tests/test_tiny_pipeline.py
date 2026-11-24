import math
import tempfile
import unittest
from pathlib import Path
from vlanix_tiny.pipeline import run

class PipelineTests(unittest.TestCase):
    def test_end_to_end(self):
        rows=[dict(text=f"Đây là bài học tiếng Việt số {i}. "+"Máy tính xử lý dữ liệu. "*(1+i%3),source="fixture",document_id=str(i),license="CC0-1.0") for i in range(24)]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"run"
            report=run(rows,out,budget=64,batch_size=8,smoke=True)
            a,b=report["arms"]["random"],report["arms"]["curriculum"]
            self.assertEqual(a["tokens"],b["tokens"])
            self.assertEqual(a["steps"],b["steps"])
            self.assertEqual(a["initial_loss"],b["initial_loss"])
            self.assertTrue(math.isfinite(a["final_loss"]))
            self.assertTrue(math.isfinite(b["final_loss"]))
            self.assertTrue(a["restore_ok"] and b["restore_ok"])
            self.assertTrue((out/"manifest.json").exists())
            self.assertTrue(report["smoke_only"])
            with self.assertRaises(FileExistsError):
                run(rows,out,budget=64,smoke=True)
    def test_real_mode_refuses_placeholder_tokenizer(self):
        rows=[dict(text=f"Bài học tiếng Việt số {i}.",source="fixture",document_id=str(i),license="CC0-1.0") for i in range(30)]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"placeholder"):
                run(rows,Path(d)/"real",budget=64)
