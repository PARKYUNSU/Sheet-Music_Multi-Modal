import json, re
from vosk import Model, KaldiRecognizer
def _tokenize_line(s: str):
    s=s.lower(); s=re.sub(r'[^0-9a-z가-힣\s]',' ',s); s=re.sub(r'\s+',' ',s).strip(); return s.split()
def build_grammar_for_lines(current_line: str, next_line: str):
    vocab=set(_tokenize_line(current_line)+_tokenize_line(next_line))
    if not vocab: vocab={'음악','가사'}
    return json.dumps(sorted(vocab), ensure_ascii=False)
class VoskEngine:
    def __init__(self, model_dir: str, samplerate: int=16000):
        self.model=Model(model_dir); self.samplerate=samplerate; self.recognizer=None
    def update_grammar(self, current_line: str, next_line: str):
        self.recognizer = KaldiRecognizer(self.model, self.samplerate, build_grammar_for_lines(current_line, next_line))
    def accept_audio(self, pcm_bytes: bytes) -> str:
        if self.recognizer is None: return ''
        if self.recognizer.AcceptWaveform(pcm_bytes):
            res=json.loads(self.recognizer.Result()); return (res.get('text') or '').strip()
        else:
            pres=json.loads(self.recognizer.PartialResult()); return (pres.get('partial') or '').strip()
