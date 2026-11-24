"""Quarantine entire held-out documents with exact cross-split chunk collisions."""
def document_key(row):
    return (row["source"],row["document_id"])

def quarantine_overlaps(by_budget):
    if not by_budget:
        raise ValueError("no budgets")
    blocked=set()
    before={}
    for budget,cs in by_budget.items():
        train_hashes={c["sha256"] for c in cs["train"]}
        collisions=train_hashes & {c["sha256"] for c in cs["validation"]}
        before[str(budget)]=len(collisions)
        blocked.update(document_key(c) for c in cs["validation"] if c["sha256"] in collisions)
    clean={b:{"train":list(cs["train"]),"validation":[c for c in cs["validation"] if document_key(c) not in blocked]} for b,cs in by_budget.items()}
    return clean,blocked,before
