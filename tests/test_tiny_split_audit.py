import unittest
from vlanix_tiny.split_audit import quarantine_overlaps

def row(doc,h,source="wiki"):
    return dict(document_id=doc,sha256=h,source=source)

class SplitAuditTests(unittest.TestCase):
    def test_one_collision_quarantines_entire_doc_across_budgets(self):
        train=[row("t","shared")]
        budgets={64:dict(train=train,validation=[row("a","shared"),row("a","unique"),row("b","safe")]),128:dict(train=[row("t","other")],validation=[row("a","joined"),row("b","safe2")])}
        clean,blocked,before=quarantine_overlaps(budgets)
        self.assertEqual(blocked,{("wiki","a")})
        self.assertEqual(before,{"64":1,"128":0})
        for b in budgets:
            self.assertEqual(clean[b]["train"],budgets[b]["train"])
            self.assertEqual([c["document_id"] for c in clean[b]["validation"]],["b"])
        self.assertEqual(len(budgets[64]["validation"]),3)
    def test_document_identity_includes_source(self):
        cs={64:dict(train=[row("t","shared")],validation=[row("a","shared"),row("a","safe","other")])}
        clean,blocked,before=quarantine_overlaps(cs)
        self.assertEqual(clean[64]["validation"],[row("a","safe","other")])
    def test_empty_validation_not_silently_repopulated(self):
        cs={64:dict(train=[row("t","shared")],validation=[row("a","shared")])}
        clean,blocked,before=quarantine_overlaps(cs)
        self.assertEqual(clean[64]["validation"],[])
