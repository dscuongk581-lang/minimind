import tempfile
import unittest
from pathlib import Path
from vlanix_tiny.tokenizer import train_tokenizer
from vlanix_tiny.chunks import chunk_document

class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tok=train_tokenizer(["Tiếng Việt có dấu. Máy tính có bộ nhớ."]*10,Path(self.tmp.name)/"tok.json")
        self.row=dict(text="Tiếng Việt có dấu. Máy tính có bộ nhớ.\n\n"+"Chuỗi dài 🙂 không mất dữ liệu. "*150,source="fixture",document_id="1",license="CC0-1.0")

    def test_all_budgets_and_exact_reconstruction(self):
        for budget in (64,128,256,512):
            chunks=chunk_document(self.row,self.tok,budget,"train")
            self.assertEqual("".join(c["text"] for c in chunks),self.row["text"])
            self.assertTrue(all(2<c["token_count"]<=budget for c in chunks))
            self.assertTrue(all(c["token_count"]==len(self.tok.encode(c["text"]).ids)+2 for c in chunks))
            self.assertEqual(len({c["chunk_id"] for c in chunks}),len(chunks))
            self.assertEqual(chunks,chunk_document(self.row,self.tok,budget,"train"))
            for c in chunks:
                self.assertEqual(self.row["text"][c["char_start"]:c["char_end"]],c["text"])
                self.assertEqual(c["license"],self.row["license"])

    def test_unbroken_unicode(self):
        row=dict(self.row,text="🙂ệ"*300)
        chunks=chunk_document(row,self.tok,64,"validation")
        self.assertEqual("".join(c["text"] for c in chunks),row["text"])
        self.assertTrue(all(c["split"]=="validation" and c["token_count"]<=64 for c in chunks))

    def test_invalid_budget_and_split(self):
        for budget,split in [(63,"train"),(64,"test")]:
            with self.assertRaises(ValueError):
                chunk_document(self.row,self.tok,budget,split)
