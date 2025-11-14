import webrtcvad
class VADGate:
    def __init__(self, samplerate=16000, aggressiveness=2, frame_ms=30, hang_ms=200):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sr = samplerate
        self.frame = int(samplerate * frame_ms / 1000)
        self.hang = hang_ms / 1000.0
        self.last_speech_t = -1.0
        self.t = 0.0
    def is_speech(self, pcm_bytes: bytes) -> bool:
        buf = memoryview(pcm_bytes); is_speech_any = False
        for i in range(0, len(buf), self.frame*2):
            chunk = buf[i:i+self.frame*2]
            if len(chunk) < self.frame*2: break
            if self.vad.is_speech(chunk.tobytes(), self.sr): is_speech_any = True
        if is_speech_any: self.last_speech_t = self.t
        self.t += len(buf) / 2 / self.sr
        return is_speech_any or (self.t - self.last_speech_t) < self.hang
