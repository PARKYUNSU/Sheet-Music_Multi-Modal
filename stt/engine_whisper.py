from faster_whisper import WhisperModel
import numpy as np
class WhisperEngine:
    def __init__(self, model_size='small', device='metal', compute_type='float16'):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
    def transcribe_chunk(self, pcm_bytes, samplerate=16000):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)/32768.0
        segments, info = self.model.transcribe(audio, language='ko', beam_size=1, vad_filter=True,
                                               vad_parameters=dict(min_silence_duration_ms=300),
                                               condition_on_previous_text=False, temperature=0.0)
        texts=[seg.text.strip() for seg in segments]
        return ' '.join([t for t in texts if t])
