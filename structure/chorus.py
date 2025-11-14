import hashlib
from collections import defaultdict
def _norm(s:str)->str: return ' '.join(s.lower().split())
def _hash_line(s:str)->str: return hashlib.md5(_norm(s).encode('utf-8')).hexdigest()
def detect_sections(lines, min_block=2, shingle=2, min_repeats=2):
    n=len(lines); h=[_hash_line(l) for l in lines]; shingles=defaultdict(list)
    for i in range(0,n-shingle+1):
        key=tuple(h[i:i+shingle]); shingles[key].append(i)
    block_candidates=[]; visited=set()
    for key,starts in shingles.items():
        if len(starts)<min_repeats: continue
        for s in starts:
            if s in visited: continue
            length=shingle
            while s+length<n:
                if s+length+shingle-1>=n: break
                k2=tuple(h[s+length:s+length+shingle])
                if k2 not in shingles or len(shingles[k2])<min_repeats: break
                length+=1
            if length>=min_block:
                block_candidates.append((s,s+length))
                for t in range(s,s+length): visited.add(t)
    block_candidates.sort(key=lambda x:(x[1]-x[0]), reverse=True)
    final_blocks=[]; occ=[False]*n
    for s,e in block_candidates:
        if any(occ[i] for i in range(s,e)): continue
        for i in range(s,e): occ[i]=True
        final_blocks.append((s,e))
    final_blocks.sort()
    labels=['']*n; blocks={}
    if not final_blocks:
        for i in range(n): labels[i]='V1'
        return labels, blocks
    block_keys=[]; freq=defaultdict(int)
    for (s,e) in final_blocks:
        k=tuple(h[s:e]); block_keys.append(k); freq[k]+=1
    chorus_key=max(freq.items(), key=lambda kv: kv[1])[0]
    chorus_ranges=[final_blocks[i] for i,k in enumerate(block_keys) if k==chorus_key]
    for s,e in chorus_ranges:
        for i in range(s,e): labels[i]='C'
    v_idx=b_idx=o_idx=1
    for (s,e),k in zip(final_blocks, block_keys):
        if k==chorus_key: continue
        tag='V'+str(v_idx) if v_idx<=2 else ('B'+str(b_idx) if b_idx<=2 else 'O'+str(o_idx))
        if v_idx<=2: v_idx+=1
        elif b_idx<=2: b_idx+=1
        else: o_idx+=1
        for i in range(s,e): labels[i]=tag
    last='V1'
    for i in range(n):
        if labels[i]: last=labels[i]
        else: labels[i]=last
    blocks['C']=chorus_ranges[0] if chorus_ranges else None
    return labels, blocks
