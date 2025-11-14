import numpy as np, librosa
class BeatEstimator:
    def __init__(self, samplerate=16000, win_seconds=6.0):
        self.sr=samplerate; self.win=int(win_seconds*samplerate); self.buf=np.zeros(self.win,dtype=np.float32)
    def update(self, pcm_bytes: bytes):
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)/32768.0
        L=len(x)
        if L>=self.win: self.buf=x[-self.win:]
        else:
            self.buf=np.roll(self.buf,-L); self.buf[-L:]=x
    def bpm_and_phase(self):
        y=self.buf.copy()
        if np.allclose(y,0.0): return None,[]
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr, units='time')
            return float(tempo), (beats.tolist() if hasattr(beats,'tolist') else list(beats))
        except Exception: return None,[]
    def early_display_hint(self, lead_ms=300):
        bpm, times = self.bpm_and_phase()
        if bpm is None or not times: return False
        interval = 60.0 / max(1e-6, bpm)
        last=times[-1]; next_t=last+interval; now=len(self.buf)/self.sr
        return (0 <= next_t - now <= (lead_ms/1000.0))
