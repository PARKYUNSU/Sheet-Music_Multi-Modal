from rapidfuzz import fuzz
from .normalizer_ko import normalize_ko

class LineMatcher:
    def __init__(self, th_lock=75, th_preview=55, th_release=50):
        self.th_lock = th_lock
        self.th_preview = th_preview
        self.th_release = th_release
        self.locked_line = -1

    def score(self, recent_text: str, target_line: str) -> float:
        a = normalize_ko(recent_text)
        b = normalize_ko(target_line)
        if not a or not b:
            return 0.0
        # partial ratio handles prefixes well
        return float(fuzz.partial_ratio(a, b))

    def decide(self, recent_text: str, lyrics: list, idx: int):
        # returns (locked_idx, show_preview:bool)
        curr = lyrics[idx] if idx < len(lyrics) else ""
        nxt  = lyrics[idx+1] if idx+1 < len(lyrics) else ""

        s_curr = self.score(recent_text, curr)
        s_next = self.score(recent_text, nxt) if nxt else 0.0

        # lock logic with hysteresis
        if self.locked_line < idx:
            if s_curr >= self.th_lock:
                self.locked_line = idx
        else:
            if s_curr < self.th_release:
                # allow unlock (rarely used to avoid jitter)
                pass

        show_preview = (self.locked_line == idx and s_next >= self.th_preview)
        return self.locked_line, show_preview
