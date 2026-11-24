import unittest
from vlanix_tiny.ingestion import collect, Source

class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.s=Source("wikimedia/wikipedia","20231101.vi","a"*40,"CC-BY-SA-3.0","review-001")
    def test_limits_and_provenance(self):
        rows=[dict(id=str(i),text="Tiếng Việt.",url="https://vi.wikipedia.org/wiki/Test") for i in range(5)]
        out,m=collect(iter(rows),self.s,2,10000)
        self.assertEqual(len(out),2)
        self.assertEqual(m["scanned"],2)
        self.assertIn("a"*40,out[0]["source"])
        self.assertEqual(out[0]["license"],"CC-BY-SA-3.0")
        self.assertEqual(out[0]["document_id"],"0")
    def test_byte_limit(self):
        out,m=collect(iter([dict(id="1",text="Tiếng Việt.",url="https://vi.wikipedia.org/wiki/Test")]),self.s,5,2)
        self.assertEqual(out,[])
        self.assertEqual(m["stop"],"byte_budget")
    def test_invalid_and_duplicate(self):
        row=dict(id="1",text="Tiếng Việt.",url="https://vi.wikipedia.org/wiki/Test")
        out,m=collect(iter([{},row,row]),self.s,3,10000)
        self.assertEqual(len(out),1)
        self.assertEqual(m["rejected"],2)
    def test_unpinned_revision_rejected(self):
        with self.assertRaises(ValueError):
            Source("wikimedia/wikipedia","20231101.vi","main","CC-BY-SA-3.0","review-001")
