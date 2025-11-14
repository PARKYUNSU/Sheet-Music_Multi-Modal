from typing import List

def write_openfst_txt(lines: List[str], path_txt: str, add_self_loops: bool=True):
    isyms = {"<eps>":0}
    osyms = {"<eps>":0}
    def sym_add(tbl, s):
        if s not in tbl:
            tbl[s] = len(tbl)
        return tbl[s]
    arcs = []
    s = 0
    for i, line in enumerate(lines):
        t = i+1
        lbl = line.strip() if line.strip() else f"LINE_{i}"
        i_id = sym_add(isyms, lbl); o_id = sym_add(osyms, lbl)
        arcs.append((s, t, i_id, o_id, 0.0))
        if add_self_loops:
            nid = sym_add(isyms, "<noise>"); nod = sym_add(osyms, "<noise>")
            arcs.append((s, s, nid, nod, 0.5))
        s = t
    final_state = len(lines)
    with open(path_txt, "w", encoding="utf-8") as f:
        for (a,b,c,d,w) in arcs:
            f.write(f"{a} {b} {c} {d} {w}\n")
        f.write(f"{final_state}\n")
    with open(path_txt.replace(".txt",".isyms.txt"), "w", encoding="utf-8") as fi:
        for s, i in sorted(isyms.items(), key=lambda kv: kv[1]):
            fi.write(f"{s} {i}\n")
    with open(path_txt.replace(".txt",".osyms.txt"), "w", encoding="utf-8") as fo:
        for s, i in sorted(osyms.items(), key=lambda kv: kv[1]):
            fo.write(f"{s} {i}\n")
