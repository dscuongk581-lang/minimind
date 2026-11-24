import tempfile
import unittest
from pathlib import Path
from vlanix_tiny.preflight import inspect_rows, REVISION

class PreflightTests(unittest.TestCase):
    def test_report_is_metadata_only(self):
        rows=[dict(id=str(i),url=f"https://vi.wikipedia.org/wiki/Test_{i}",text=f"Bài học tiếng Việt số {i}. Máy tính xử lý dữ liệu.") for i in range(40)]
        with tempfile.TemporaryDirectory() as d:
            report=inspect_rows(iter(rows),Path(d),40,100000,64)
            self.assertEqual(report["source"]["revision"],REVISION)
            self.assertEqual(report["ingestion"]["accepted"],40)
            self.assertGreater(report["train_documents"],0)
            self.assertGreater(report["validation_documents"],0)
            self.assertFalse(report["training_authorized"])
            self.assertGreater(report["reserved_tokens"],0)
            self.assertFalse(any(Path(d).iterdir()))
            self.assertNotIn("Bài học",str(report))
