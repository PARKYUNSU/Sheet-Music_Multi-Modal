import re
_punc=re.compile(r'[\.,!?~\-\(\)\[\]\{\}\"\'\:;]'); _spaces=re.compile(r'\s+')
HANGUL_BASE=0xAC00; CHOSUNG=588; JWUNGSUNG=28
def decompose_korean_syllable(ch):
    code=ord(ch)
    if 0xAC00<=code<=0xD7A3:
        s_index=code-HANGUL_BASE; cho=s_index//CHOSUNG; jung=(s_index%CHOSUNG)//JWUNGSUNG; jong=s_index%JWUNGSUNG
        return cho,jung,jong
    return None
def collapse_singing_stretch(s:str)->str:
    if not s: return s
    out=[]; prev_syll=None; prev_jung=None; run_syll=0; run_jung=0
    for ch in s:
        if ch.isspace(): out.append(ch); prev_syll=None; prev_jung=None; run_syll=0; run_jung=0; continue
        dec=decompose_korean_syllable(ch)
        if dec is None: out.append(ch); prev_syll=None; prev_jung=None; run_syll=0; run_jung=0; continue
        _,jung,_=dec
        if prev_syll==ch:
            run_syll+=1
            if run_syll<=2: out.append(ch)
            continue
        else: run_syll=1
        if prev_jung is not None and jung==prev_jung:
            run_jung+=1
            if run_jung<=1: out.append(ch)
        else: run_jung=0; out.append(ch)
        prev_syll=ch; prev_jung=jung
    return ''.join(out)
def normalize_ko(s:str)->str:
    s=s.lower(); s=_punc.sub(' ',s); s=collapse_singing_stretch(s); s=_spaces.sub(' ',s).strip(); return s
