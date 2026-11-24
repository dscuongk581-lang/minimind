"""Explicit operator review attestation and multi-budget split audit."""
import hashlib
import json
from vlanix_tiny.split_audit import quarantine_overlaps

def review_digest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()

def check_review(rows,receipt):
    if not isinstance(receipt,dict) or receipt.get("decision")!="approved-internal-research" or receipt.get("input_sha256")!=review_digest(rows) or not isinstance(receipt.get("review_id"),str) or not receipt["review_id"].strip():
        raise ValueError("operator source review receipt missing or mismatched")
    return receipt["review_id"]

def audit_chunks(raw):
    clean,blocked,before=quarantine_overlaps(raw)
    after={str(b):len({c["sha256"] for c in cs["train"]}&{c["sha256"] for c in cs["validation"]}) for b,cs in clean.items()}
    if any(after.values()) or any(not cs["validation"] for cs in clean.values()):
        raise ValueError("audited validation unavailable")
    return clean,dict(quarantined_documents=len(blocked),before=before,after=after,blocked_documents=[list(x) for x in sorted(blocked)])
