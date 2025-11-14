#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP3 파일을 재생하면서 실시간으로 가사를 인식하고 표시하는 프로그램
"""
import sys, os, time, argparse
from collections import deque
import numpy as np
import sounddevice as sd
import soundfile as sf

# 프로젝트 루트를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from stt.engine_whisper import WhisperEngine
from align.matcher import LineMatcher, tail_token_coverage
from align.normalizer_ko import normalize_ko

def load_lyrics(path='lyrics.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]

def play_mp3_with_lyrics(mp3_path, lyrics_path, model='small', device='cpu', chunk_sec=3.0, history_sec=6):
    """MP3 파일을 재생하면서 실시간으로 가사를 인식"""
    
    print(f"[MP3 가사 플레이어]")
    print(f"MP3: {mp3_path}")
    print(f"가사: {lyrics_path}")
    print(f"모델: {model}, 디바이스: {device}")
    print("-" * 60)
    
    # 가사 로드
    lyrics = load_lyrics(lyrics_path)
    if not lyrics:
        print("[오류] 가사 파일이 비어있습니다.")
        return
    
    print(f"총 {len(lyrics)}줄 가사 로드 완료\n")
    
    # STT 엔진 초기화
    print("Whisper 모델 로딩 중...")
    stt = WhisperEngine(model_size=model, device=device, compute_type='int8')
    print("모델 로딩 완료!\n")
    
    # 매칭 엔진 초기화
    matcher = LineMatcher(th_lock=75, th_preview=55, th_release=50)
    
    # MP3 파일 로드
    print(f"MP3 파일 로딩 중: {mp3_path}")
    audio, sr = sf.read(mp3_path, dtype='float32')
    print(f"샘플레이트: {sr} Hz, 길이: {len(audio)/sr:.1f}초")
    
    # 스테레오 -> 모노
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # 16kHz로 리샘플링
    target_sr = 16000
    if sr != target_sr:
        print(f"리샘플링 중: {sr} Hz -> {target_sr} Hz")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # 청크 크기 계산
    chunk_samples = int(sr * chunk_sec)
    
    # 오디오 재생 준비
    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(f"[재생 오류] {status}")
    
    # 변수 초기화
    recent = deque(maxlen=int(history_sec / chunk_sec))
    line_idx = 0
    locked_idx = -1
    
    print("\n" + "=" * 60)
    print("🎵 재생 시작!")
    print("=" * 60 + "\n")
    
    # 오디오 스트림 시작 (재생)
    with sd.OutputStream(samplerate=sr, channels=1, callback=audio_callback):
        # 청크 단위로 처리
        for i in range(0, len(audio), chunk_samples):
            if line_idx >= len(lyrics):
                break
            
            # 현재 청크 추출
            chunk = audio[i:i+chunk_samples]
            if len(chunk) < chunk_samples:
                # 마지막 청크는 패딩
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            # 현재 재생 시간
            current_time = i / sr
            
            # 청크 재생
            sd.play(chunk, samplerate=sr)
            
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
                
                # 다음 줄 점수 (미리보기)
                score_next = 0
                if line_idx + 1 < len(lyrics):
                    score_next = matcher.score(recent_text, lyrics[line_idx + 1])
                
                # 콘솔 출력
                print(f"\r[{current_time:6.1f}s] STT: {text[:40]:40s} | 점수: {score_curr:5.1f}", end='')
                
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
                
                # 미리보기 표시
                elif show_preview:
                    if line_idx + 1 < len(lyrics) and score_next > matcher.th_preview:
                        print(f"\n💡 [미리보기] 다음 줄: {lyrics[line_idx + 1][:30]}...")
            
            # 청크 재생 완료 대기
            sd.wait()
    
    print("\n" + "=" * 60)
    print(f"🎉 재생 완료! (총 {line_idx}/{len(lyrics)}줄 인식)")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='MP3 파일 가사 동기화 플레이어')
    parser.add_argument('--mp3', default='주 품에 품으소서.mp3', help='MP3 파일 경로')
    parser.add_argument('--lyrics', default='lyrics.txt', help='가사 파일 경로')
    parser.add_argument('--model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large'], help='Whisper 모델 크기')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'], help='디바이스')
    parser.add_argument('--chunk', type=float, default=3.0, help='청크 길이 (초)')
    parser.add_argument('--history', type=float, default=6.0, help='히스토리 윈도우 (초)')
    
    args = parser.parse_args()
    
    try:
        play_mp3_with_lyrics(
            mp3_path=args.mp3,
            lyrics_path=args.lyrics,
            model=args.model,
            device=args.device,
            chunk_sec=args.chunk,
            history_sec=args.history
        )
    except KeyboardInterrupt:
        print("\n\n중단됨.")
    except Exception as e:
        print(f"\n[오류] {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


