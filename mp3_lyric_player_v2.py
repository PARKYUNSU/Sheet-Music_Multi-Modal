#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP3 파일을 재생하면서 실시간으로 가사를 인식하고 표시하는 프로그램 (개선 버전)
재생과 STT 처리를 분리하여 소리 깨짐 방지
"""
import sys, os, time, argparse, threading
from collections import deque
import numpy as np
import soundfile as sf

# 프로젝트 루트를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from stt.engine_whisper import WhisperEngine
from align.matcher import LineMatcher, tail_token_coverage

# pygame을 사용한 안정적인 재생
try:
    import pygame
    USE_PYGAME = True
except ImportError:
    import sounddevice as sd
    USE_PYGAME = False
    print("[경고] pygame이 없어 sounddevice를 사용합니다. 소리가 깨질 수 있습니다.")

def load_lyrics(path='lyrics.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]

class MP3LyricPlayer:
    def __init__(self, mp3_path, lyrics_path, model='small', device='cpu', chunk_sec=3.0, history_sec=6):
        self.mp3_path = mp3_path
        self.lyrics_path = lyrics_path
        self.model = model
        self.device = device
        self.chunk_sec = chunk_sec
        self.history_sec = history_sec
        
        self.playing = False
        self.current_time = 0.0
        self.audio = None
        self.sr = 16000
        
    def load_audio(self):
        """오디오 파일 로드 및 전처리"""
        print(f"MP3 파일 로딩 중: {self.mp3_path}")
        audio, sr = sf.read(self.mp3_path, dtype='float32')
        print(f"샘플레이트: {sr} Hz, 길이: {len(audio)/sr:.1f}초")
        
        # 스테레오 -> 모노
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # 16kHz로 리샘플링
        if sr != 16000:
            print(f"리샘플링 중: {sr} Hz -> 16000 Hz")
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        self.audio = audio
        self.sr = sr
        return audio, sr
    
    def play_thread(self):
        """별도 스레드에서 오디오 재생"""
        if USE_PYGAME:
            # pygame 사용
            pygame.mixer.init(frequency=self.sr, channels=1)
            pygame.mixer.music.load(self.mp3_path)
            pygame.mixer.music.play()
            
            while self.playing and pygame.mixer.music.get_busy():
                self.current_time = pygame.mixer.music.get_pos() / 1000.0
                time.sleep(0.1)
        else:
            # sounddevice 사용 (블로킹 재생)
            import sounddevice as sd
            sd.play(self.audio, self.sr)
            start_time = time.time()
            while self.playing:
                self.current_time = time.time() - start_time
                if self.current_time >= len(self.audio) / self.sr:
                    break
                time.sleep(0.1)
            sd.stop()
    
    def run(self):
        """메인 실행"""
        print(f"\n[MP3 가사 플레이어 v2]")
        print(f"MP3: {self.mp3_path}")
        print(f"가사: {self.lyrics_path}")
        print(f"모델: {self.model}, 디바이스: {self.device}")
        print("-" * 60)
        
        # 가사 로드
        lyrics = load_lyrics(self.lyrics_path)
        if not lyrics:
            print("[오류] 가사 파일이 비어있습니다.")
            return
        
        print(f"총 {len(lyrics)}줄 가사 로드 완료")
        for i, line in enumerate(lyrics):
            print(f"  {i+1}. {line}")
        print()
        
        # 오디오 로드
        audio, sr = self.load_audio()
        
        # STT 엔진 초기화
        print("Whisper 모델 로딩 중...")
        stt = WhisperEngine(model_size=self.model, device=self.device, compute_type='int8')
        print("모델 로딩 완료!\n")
        
        # 매칭 엔진 초기화
        matcher = LineMatcher(th_lock=75, th_preview=55, th_release=50)
        
        # 청크 크기 계산
        chunk_samples = int(sr * self.chunk_sec)
        
        # 변수 초기화
        recent = deque(maxlen=int(self.history_sec / self.chunk_sec))
        line_idx = 0
        
        print("\n" + "=" * 60)
        print("🎵 재생 시작!")
        print("=" * 60 + "\n")
        
        # 재생 스레드 시작
        self.playing = True
        play_thread = threading.Thread(target=self.play_thread, daemon=True)
        play_thread.start()
        
        time.sleep(0.5)  # 재생 시작 대기
        
        # 청크 단위로 처리
        try:
            for i in range(0, len(audio), chunk_samples):
                if line_idx >= len(lyrics):
                    break
                
                if not self.playing:
                    break
                
                # 현재 청크 추출
                chunk = audio[i:i+chunk_samples]
                if len(chunk) < chunk_samples:
                    # 마지막 청크는 패딩
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                
                # PCM 변환 (STT 입력용)
                pcm_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                
                # STT 인식
                text = stt.transcribe_chunk(pcm_bytes, samplerate=sr)
                
                if text:
                    recent.append(text)
                    recent_text = ' '.join(list(recent)[-3:])  # 최근 3개 청크
                    
                    # 가사 매칭
                    locked_idx, show_preview = matcher.decide(recent_text, lyrics, line_idx)
                    
                    # 현재 줄 점수
                    score_curr = matcher.score(recent_text, lyrics[line_idx])
                    
                    # 콘솔 출력 (현재 시간 표시)
                    chunk_time = i / sr
                    print(f"\r[{chunk_time:6.1f}s] 인식: {text[:50]:50s} | 점수: {score_curr:5.1f}", end='', flush=True)
                    
                    # 줄 전환 조건
                    if locked_idx == line_idx:
                        cover = tail_token_coverage(recent_text, lyrics[line_idx], tail_ratio=0.5)
                        should_advance = (score_curr >= 88) or (cover >= 0.6)
                        
                        if should_advance:
                            print(f"\n{'=' * 60}")
                            print(f"✅ [{line_idx + 1}/{len(lyrics)}] {lyrics[line_idx]}")
                            print(f"   인식: {text}")
                            print(f"   점수: {score_curr:.1f}, 커버리지: {cover:.1%}")
                            print(f"{'=' * 60}\n")
                            line_idx += 1
                
                # 청크 처리 완료 대기 (실제 재생 시간에 맞춤)
                expected_time = (i + chunk_samples) / sr
                while self.current_time < expected_time and self.playing:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n중단됨.")
        finally:
            self.playing = False
            if USE_PYGAME:
                pygame.mixer.music.stop()
            
            print("\n" + "=" * 60)
            print(f"🎉 재생 완료! (총 {line_idx}/{len(lyrics)}줄 인식)")
            print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='MP3 파일 가사 동기화 플레이어 v2')
    parser.add_argument('--mp3', default='주 품에 품으소서.mp3', help='MP3 파일 경로')
    parser.add_argument('--lyrics', default='lyrics.txt', help='가사 파일 경로')
    parser.add_argument('--model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large'], help='Whisper 모델 크기')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'], help='디바이스')
    parser.add_argument('--chunk', type=float, default=3.0, help='청크 길이 (초)')
    parser.add_argument('--history', type=float, default=6.0, help='히스토리 윈도우 (초)')
    
    args = parser.parse_args()
    
    player = MP3LyricPlayer(
        mp3_path=args.mp3,
        lyrics_path=args.lyrics,
        model=args.model,
        device=args.device,
        chunk_sec=args.chunk,
        history_sec=args.history
    )
    
    player.run()

if __name__ == '__main__':
    main()


