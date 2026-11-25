"""Consolidation data and strict parent eligibility for the finite core."""
import re
from vlanix_tiny.consolidation import fresh_frames


def third_frames(train,fresh,held,seed):
    if len(set(fresh))!=len(fresh) or len(set(held))!=len(held) or set(fresh)&set(held):
        raise ValueError('overlap')
    return fresh_frames(train,list(fresh)+list(held),seed)


def validate_parent(row,seed,digest):
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('digest')
    if row.get('seed')!=seed or row.get('status')!='completed' or row.get('local_correct')!=32 or type(row.get('finite_gate')) is not bool or row.get('checkpoint_sha256')!=digest:
        raise ValueError('parent identity')
    test=row.get('test_cached',{})
    if row.get('test_rows_used')!=832 or test.get('rows')!=832 or test.get('correct')!=832:
        raise ValueError('parent test')
    ops=row.get('exhaustive',{})
    if set(ops)!=set(('add','xor','and','or')) or any(g.get('rows')!=65536 or g.get('correct')!=65536 for g in ops.values()):
        raise ValueError('parent arithmetic')
