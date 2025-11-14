import queue, numpy as np, sounddevice as sd
try:
    import librosa
except Exception:
    librosa = None
def _stereo_to_mono(x):
    if x.ndim == 1: return x
    return np.mean(x, axis=1)
def _resample(x, sr_from, sr_to):
    if sr_from == sr_to: return x
    if librosa is not None:
        return librosa.resample(x.astype(np.float32), orig_sr=sr_from, target_sr=sr_to)
    ratio = sr_to / sr_from
    n_new = int(round(len(x) * ratio))
    xp = np.linspace(0, 1, len(x), endpoint=False)
    xq = np.linspace(0, 1, n_new, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)
class AudioStream:
    def __init__(self, samplerate=16000, block_duration=1.0, channels=1, device_name=None, device_index=None, native_rate=None):
        self.target_sr = samplerate
        self.block_dur = block_duration
        self.channels = channels
        self.device = device_index if device_index is not None else device_name
        self.native_rate = native_rate
        self.block_size = int((self.native_rate or self.target_sr) * block_duration)
        self.q = queue.Queue()
        self.stream = None
    def _callback(self, indata, frames, time, status):
        if status: print('[audio] status:', status)
        x = indata.copy()
        x = _stereo_to_mono(x)
        sr_in = int(self.stream.samplerate)
        if sr_in != self.target_sr:
            x = _resample(x, sr_in, self.target_sr)
        pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        self.q.put(pcm)
    def start(self):
        self.stream = sd.InputStream(device=self.device, channels=max(1, self.channels),
                                     samplerate=self.native_rate, blocksize=self.block_size,
                                     callback=self._callback, dtype='float32')
        self.stream.start()
    def read(self): return self.q.get()
    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close()
