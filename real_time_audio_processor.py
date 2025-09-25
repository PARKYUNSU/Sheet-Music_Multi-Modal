"""
실시간 오디오 처리 및 음높이 검출 시스템
MP3 파일을 입력으로 하여 실시간 오디오 분석 수행
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pyaudio
import threading
import time
from collections import deque
import json
from typing import List, Dict, Optional, Tuple

class RealTimeAudioProcessor:
    def __init__(self, audio_file_path: str, chunk_size: int = 1024, sample_rate: int = 22050):
        """
        실시간 오디오 처리기 초기화
        
        Args:
            audio_file_path: MP3 파일 경로
            chunk_size: 청크 크기 (프레임 단위)
            sample_rate: 샘플링 레이트
        """
        self.audio_file_path = audio_file_path
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        
        # 오디오 데이터 로드
        self.audio_data = None
        self.duration = 0
        self.load_audio_file()
        
        # 실시간 처리 변수
        self.is_playing = False
        self.current_position = 0
        self.pitch_history = deque(maxlen=100)  # 최근 100개 음높이 저장
        self.volume_history = deque(maxlen=100)  # 최근 100개 볼륨 저장
        
        # PyAudio 설정
        self.p = None
        self.stream = None
        
        # 음높이 검출 설정
        self.pitch_threshold = 0.1  # 음높이 검출 최소 임계값
        self.f0_min = 80.0  # 최소 주파수 (Hz)
        self.f0_max = 1000.0  # 최대 주파수 (Hz)
        
        print(f"🎵 실시간 오디오 처리기 초기화 완료")
        print(f"   - 파일: {audio_file_path}")
        print(f"   - 길이: {self.duration:.2f}초")
        print(f"   - 샘플링 레이트: {self.sample_rate}Hz")
    
    def load_audio_file(self):
        """MP3 파일 로드"""
        try:
            print(f"📁 오디오 파일 로딩 중: {self.audio_file_path}")
            
            # librosa로 오디오 로드
            self.audio_data, sr = librosa.load(
                self.audio_file_path, 
                sr=self.sample_rate,
                mono=True  # 모노로 변환
            )
            
            self.duration = len(self.audio_data) / self.sample_rate
            print(f"✅ 오디오 로드 완료: {len(self.audio_data)} 샘플, {self.duration:.2f}초")
            
        except Exception as e:
            print(f"❌ 오디오 파일 로드 실패: {e}")
            raise
    
    def detect_pitch(self, audio_chunk: np.ndarray) -> Tuple[float, float]:
        """
        음높이 검출 (YIN 알고리즘 사용)
        
        Args:
            audio_chunk: 오디오 청크
            
        Returns:
            (음높이(Hz), 신뢰도)
        """
        try:
            # YIN 알고리즘으로 음높이 검출 (threshold 매개변수 제거)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_chunk,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sample_rate
            )
            
            # 유효한 음높이만 추출
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                # 평균 음높이 계산
                avg_f0 = np.mean(valid_f0)
                confidence = np.mean(voiced_probs[~np.isnan(voiced_probs)])
                return avg_f0, confidence
            else:
                return 0.0, 0.0
                
        except Exception as e:
            print(f"음높이 검출 오류: {e}")
            return 0.0, 0.0
    
    def detect_volume(self, audio_chunk: np.ndarray) -> float:
        """볼륨 검출 (RMS)"""
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def frequency_to_note(self, frequency: float) -> str:
        """주파수를 음표명으로 변환"""
        if frequency <= 0:
            return "Silence"
        
        # A4 = 440Hz를 기준으로 계산
        A4 = 440.0
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # 반음 단위로 변환
        semitones = 12 * np.log2(frequency / A4)
        
        # 옥타브와 음표 계산
        octave = int(4 + semitones // 12)
        note_index = int(semitones % 12)
        
        if note_index < 0:
            note_index += 12
            octave -= 1
        
        return f"{note_names[note_index]}{octave}"
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """오디오 청크 처리"""
        # 음높이 검출
        pitch, confidence = self.detect_pitch(audio_chunk)
        
        # 볼륨 검출
        volume = self.detect_volume(audio_chunk)
        
        # 음표명 변환
        note_name = self.frequency_to_note(pitch)
        
        # 히스토리에 추가
        self.pitch_history.append(pitch)
        self.volume_history.append(volume)
        
        return {
            'pitch': pitch,
            'confidence': confidence,
            'volume': volume,
            'note_name': note_name,
            'timestamp': time.time()
        }
    
    def play_audio_realtime(self, callback_interval: float = 0.1):
        """
        실시간 오디오 재생 및 분석
        
        Args:
            callback_interval: 콜백 호출 간격 (초)
        """
        print(f"🎵 실시간 오디오 재생 시작...")
        
        # PyAudio 초기화
        self.p = pyaudio.PyAudio()
        
        # 스트림 설정
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.is_playing = True
        self.current_position = 0
        
        # 오디오 재생 및 분석 스레드
        def audio_thread():
            chunk_samples = int(self.chunk_size)
            callback_samples = int(self.sample_rate * callback_interval)
            
            while self.is_playing and self.current_position < len(self.audio_data):
                # 현재 청크 추출
                end_pos = min(self.current_position + chunk_samples, len(self.audio_data))
                audio_chunk = self.audio_data[self.current_position:end_pos]
                
                # 오디오 재생
                if len(audio_chunk) > 0:
                    self.stream.write(audio_chunk.astype(np.float32).tobytes())
                
                # 오디오 분석
                if len(audio_chunk) >= callback_samples:
                    analysis_chunk = audio_chunk[:callback_samples]
                    result = self.process_audio_chunk(analysis_chunk)
                    
                    # 결과 출력
                    if result['pitch'] > 0:
                        print(f"🎵 {result['note_name']:>4} | "
                              f"주파수: {result['pitch']:6.1f}Hz | "
                              f"신뢰도: {result['confidence']:.2f} | "
                              f"볼륨: {result['volume']:.3f}")
                    else:
                        print(f"🔇 Silence | 볼륨: {result['volume']:.3f}")
                
                self.current_position = end_pos
                
                # 재생 속도 조절
                time.sleep(callback_interval)
        
        # 스레드 시작
        audio_thread_obj = threading.Thread(target=audio_thread)
        audio_thread_obj.start()
        
        return audio_thread_obj
    
    def stop_audio(self):
        """오디오 재생 중지"""
        self.is_playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        print("⏹️ 오디오 재생 중지")
    
    def analyze_full_audio(self) -> Dict:
        """전체 오디오 분석"""
        print("🔍 전체 오디오 분석 시작...")
        
        # 전체 오디오를 청크 단위로 분석
        chunk_size = int(self.sample_rate * 0.1)  # 0.1초 청크
        results = []
        
        for i in range(0, len(self.audio_data), chunk_size):
            chunk = self.audio_data[i:i+chunk_size]
            if len(chunk) > 0:
                result = self.process_audio_chunk(chunk)
                result['time'] = i / self.sample_rate
                results.append(result)
        
        # 통계 계산
        pitches = [r['pitch'] for r in results if r['pitch'] > 0]
        volumes = [r['volume'] for r in results]
        
        analysis_result = {
            'total_duration': self.duration,
            'total_chunks': len(results),
            'pitch_statistics': {
                'min_pitch': min(pitches) if pitches else 0,
                'max_pitch': max(pitches) if pitches else 0,
                'avg_pitch': np.mean(pitches) if pitches else 0,
                'pitch_range': max(pitches) - min(pitches) if pitches else 0
            },
            'volume_statistics': {
                'min_volume': min(volumes) if volumes else 0,
                'max_volume': max(volumes) if volumes else 0,
                'avg_volume': np.mean(volumes) if volumes else 0
            },
            'detailed_results': results
        }
        
        print(f"✅ 전체 오디오 분석 완료")
        print(f"   - 총 청크 수: {len(results)}")
        print(f"   - 음높이 범위: {analysis_result['pitch_statistics']['min_pitch']:.1f}Hz - {analysis_result['pitch_statistics']['max_pitch']:.1f}Hz")
        print(f"   - 평균 음높이: {analysis_result['pitch_statistics']['avg_pitch']:.1f}Hz")
        
        return analysis_result
    
    def visualize_audio_analysis(self, analysis_result: Dict):
        """오디오 분석 결과 시각화"""
        print("📊 오디오 분석 결과 시각화...")
        
        results = analysis_result['detailed_results']
        times = [r['time'] for r in results]
        pitches = [r['pitch'] for r in results]
        volumes = [r['volume'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # 그래프 생성
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 음높이 그래프
        axes[0].plot(times, pitches, 'b-', alpha=0.7)
        axes[0].set_title('음높이 변화 (Hz)')
        axes[0].set_ylabel('주파수 (Hz)')
        axes[0].grid(True, alpha=0.3)
        
        # 볼륨 그래프
        axes[1].plot(times, volumes, 'r-', alpha=0.7)
        axes[1].set_title('볼륨 변화')
        axes[1].set_ylabel('볼륨')
        axes[1].grid(True, alpha=0.3)
        
        # 신뢰도 그래프
        axes[2].plot(times, confidences, 'g-', alpha=0.7)
        axes[2].set_title('음높이 검출 신뢰도')
        axes[2].set_ylabel('신뢰도')
        axes[2].set_xlabel('시간 (초)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('audio_analysis_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 시각화 완료: audio_analysis_result.png")
    
    def save_analysis_result(self, analysis_result: Dict, filename: str = 'audio_analysis.json'):
        """분석 결과 저장"""
        # NumPy 타입을 JSON 직렬화 가능하도록 변환
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        serializable_result = convert_numpy_types(analysis_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 분석 결과 저장: {filename}")

def main():
    """메인 함수"""
    print("🎵 실시간 오디오 처리 시스템 시작")
    
    # 오디오 파일 경로
    audio_file = "주 품에 품으소서.mp3"
    
    try:
        # 오디오 처리기 초기화
        processor = RealTimeAudioProcessor(audio_file)
        
        # 전체 오디오 분석
        analysis_result = processor.analyze_full_audio()
        
        # 결과 시각화
        processor.visualize_audio_analysis(analysis_result)
        
        # 결과 저장
        processor.save_analysis_result(analysis_result)
        
        # 실시간 재생 옵션
        print("\n🎵 실시간 재생을 시작하시겠습니까? (y/n)")
        choice = input().lower()
        
        if choice == 'y':
            print("실시간 재생 시작... (Ctrl+C로 중지)")
            try:
                audio_thread = processor.play_audio_realtime()
                audio_thread.join()
            except KeyboardInterrupt:
                processor.stop_audio()
                print("\n⏹️ 사용자에 의해 중지됨")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
