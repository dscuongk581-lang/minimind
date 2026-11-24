import unittest
from vlanix_tiny.corpus import prepare

def row(text, doc="a", **kw):
    return dict(text=text, document_id=doc, source="original-fixture", license="CC0-1.0", **kw)

class CorpusTests(unittest.TestCase):
    def test_normalization_and_dedup(self):
        a,b,m=prepare([row("Tiếng Việt là ngôn ngữ đẹp."),row("Tie\u0301ng Việt là ngôn ngữ đẹp.","b")])
        self.assertEqual(len(a)+len(b),1)
        self.assertEqual(m["duplicates"],1)

    def test_document_isolation_and_order(self):
        rows=[row(f"Đây là đoạn văn tiếng Việt số {i}.",str(i//2)) for i in range(100)]
        a,b,m=prepare(rows)
        self.assertTrue(a and b)
        self.assertFalse({x["document_id"] for x in a}&{x["document_id"] for x in b})
        self.assertEqual((a,b,m),prepare(reversed(rows)))

    def test_reject_invalid_records(self):
        rows=[row("Tiếng Việt có dấu thanh."),row(""),row("Tiếng Việt <|tool_call|> nguy hiểm."),dict(row("Nội dung tiếng Việt hợp lệ."),license="unknown"),row("Contact test@example.com please."),row("Tiếng Việt chứa\x00 ký tự lỗi.")]
        a,b,m=prepare(rows)
        self.assertEqual(len(a)+len(b),1)
        self.assertEqual(sum(m["rejected"].values()),5)

    def test_duplicate_documents_cannot_leak(self):
        rows=[row("Câu tiếng Việt giống nhau ở hai tài liệu.","a"),row("Câu tiếng Việt giống nhau ở hai tài liệu.","b"),row("Đoạn văn khác của tài liệu thứ nhất.","a"),row("Đoạn văn khác của tài liệu thứ hai.","b")]
        a,b,m=prepare(rows)
        self.assertFalse(a and b)
        self.assertEqual(len(a)+len(b),3)
