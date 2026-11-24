import unittest
from vlanix_tiny.pilot_gate import audit_chunks, check_review, review_digest

def chunk(doc,h,n=8):
    return dict(source="wiki",document_id=doc,sha256=h,token_count=n)

class PilotGateTests(unittest.TestCase):
    def test_audit_keeps_same_train_and_removes_whole_validation_document(self):
        raw={64:dict(train=[chunk("t","x")],validation=[chunk("v","x"),chunk("w","y")]),128:dict(train=[chunk("t","z")],validation=[chunk("v","a"),chunk("w","b")])}
        clean,report=audit_chunks(raw)
        self.assertEqual(report["quarantined_documents"],1)
        self.assertEqual(report["before"],{"64":1,"128":0})
        self.assertEqual(report["after"],{"64":0,"128":0})
        self.assertEqual(clean[128]["train"],raw[128]["train"])
        self.assertEqual(clean[128]["validation"],[chunk("w","b")])
    def test_review_binds_exact_data_and_requires_reference(self):
        rows=[dict(source="wiki",document_id="1",text="test",license="CC-BY-SA-3.0")]
        receipt=dict(input_sha256=review_digest(rows),review_id="operator-review-1",decision="approved-internal-research")
        self.assertEqual(check_review(rows,receipt),receipt["review_id"])
        for invalid in (None,dict(receipt,review_id=""),dict(receipt,input_sha256="0"*64),dict(receipt,decision="pending")):
            with self.assertRaises(ValueError):
                check_review(rows,invalid)
        with self.assertRaises(ValueError):
            check_review([dict(rows[0],text="changed")],receipt)
