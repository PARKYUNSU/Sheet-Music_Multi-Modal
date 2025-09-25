"""
악보-연주 싱크 시스템
OMR 결과와 실시간 오디오를 동기화하여 현재 연주 위치를 추적
"""

import json
import numpy as np
import time
import threading
from typing import List, Dict, Optional, Tuple
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
from improved_omr import ImprovedOMR
from real_time_audio_processor import RealTimeAudioProcessor

class SheetMusicSyncSystem:
    def __init__(self, sheet_music_path: str, audio_file_path: str):
        """
        악보-연주 싱크 시스템 초기화
        
        Args:
            sheet_music_path: 악보 이미지 경로
            audio_file_path: 오디오 파일 경로
        """
        self.sheet_music_path = sheet_music_path
        self.audio_file_path = audio_file_path
        
        # OMR 시스템 초기화
        print("🎼 OMR 시스템 초기화...")
        self.omr = ImprovedOMR()
        self.sheet_music_data = None
        
        # 오디오 처리 시스템 초기화
        print("🎵 오디오 처리 시스템 초기화...")
        self.audio_processor = RealTimeAudioProcessor(audio_file_path)
        
        # 동기화 상태
        self.sync_state = {
            'current_position': 0,  # 현재 연주 위치 (음표 인덱스)
            'sync_confidence': 0.0,  # 동기화 신뢰도 (0-1)
            'is_synced': False,  # 동기화 상태
            'tempo': 120.0,  # 현재 템포 (BPM)
            'last_sync_time': 0,  # 마지막 동기화 시간
            'sync_history': deque(maxlen=50)  # 동기화 히스토리
        }
        
        # 매칭 설정
        self.pitch_tolerance = 50  # 음높이 허용 오차 (센트)
        self.tempo_tolerance = 0.2  # 템포 허용 오차 (20%)
        self.sync_threshold = 0.7  # 동기화 최소 신뢰도
        
        print("✅ 악보-연주 싱크 시스템 초기화 완료")
    
    def load_sheet_music(self):
        """악보 데이터 로드 및 전처리"""
        print("📄 악보 데이터 로딩...")
        
        # OMR로 악보 분석
        self.sheet_music_data = self.omr.process_sheet_music_improved(self.sheet_music_path)
        
        # 음표 데이터를 시간축으로 변환
        self.convert_notes_to_timeline()
        
        print(f"✅ 악보 로드 완료: {len(self.sheet_music_data['notes'])}개 음표")
    
    def convert_notes_to_timeline(self):
        """음표 데이터를 시간축으로 변환"""
        if not self.sheet_music_data or 'notes' not in self.sheet_music_data:
            return
        
        notes = self.sheet_music_data['notes']
        tempo = self.sync_state['tempo']
        
        # 각 음표에 시간 정보 추가
        current_time = 0.0
        for i, note in enumerate(notes):
            # 음표 시작 시간
            note['start_time'] = current_time
            
            # 음표 길이 계산 (duration을 초 단위로 변환)
            note_duration = note.get('duration', 1.0)  # 기본 1박자
            note['end_time'] = current_time + (note_duration * 60.0 / tempo)
            
            # 다음 음표 시작 시간
            current_time = note['end_time']
            
            # 음표 인덱스 추가
            note['index'] = i
        
        print(f"✅ 시간축 변환 완료: 총 {current_time:.2f}초")
    
    def frequency_to_cents(self, freq1: float, freq2: float) -> float:
        """두 주파수 간의 센트 차이 계산"""
        if freq1 <= 0 or freq2 <= 0:
            return float('inf')
        return 1200 * np.log2(freq2 / freq1)
    
    def note_name_to_frequency(self, note_name: str) -> float:
        """음표명을 주파수로 변환"""
        if note_name == "Silence":
            return 0.0
        
        # A4 = 440Hz 기준
        note_map = {
            'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
            'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        }
        
        try:
            note = note_name[:-1]  # 음표명 (C, D, E 등)
            octave = int(note_name[-1])  # 옥타브 (4, 5 등)
            
            semitones = note_map[note] + (octave - 4) * 12
            frequency = 440.0 * (2 ** (semitones / 12))
            return frequency
        except:
            return 0.0
    
    def match_pitch(self, current_pitch: float, note_frequency: float) -> float:
        """음높이 매칭 점수 계산"""
        if current_pitch <= 0 or note_frequency <= 0:
            return 0.0
        
        cents_diff = abs(self.frequency_to_cents(current_pitch, note_frequency))
        
        # 허용 오차 내에 있으면 높은 점수
        if cents_diff <= self.pitch_tolerance:
            return 1.0 - (cents_diff / self.pitch_tolerance)
        else:
            return 0.0
    
    def find_best_match(self, current_pitch: float, current_time: float) -> Tuple[Optional[Dict], float]:
        """현재 연주와 가장 잘 맞는 악보 위치 찾기"""
        if not self.sheet_music_data or 'notes' not in self.sheet_music_data:
            return None, 0.0
        
        notes = self.sheet_music_data['notes']
        best_match = None
        best_score = 0.0
        
        # 현재 시간 주변의 음표들을 검사
        time_window = 5.0  # 5초 윈도우
        
        for note in notes:
            note_freq = self.note_name_to_frequency(note['pitch'])
            
            # 시간 매칭 점수
            time_diff = abs(current_time - note['start_time'])
            if time_diff > time_window:
                continue
            
            time_score = 1.0 - (time_diff / time_window)
            
            # 음높이 매칭 점수
            pitch_score = self.match_pitch(current_pitch, note_freq)
            
            # 종합 점수 (시간 60%, 음높이 40%)
            total_score = 0.6 * time_score + 0.4 * pitch_score
            
            if total_score > best_score:
                best_score = total_score
                best_match = note
        
        return best_match, best_score
    
    def update_sync_state(self, current_pitch: float, current_time: float):
        """동기화 상태 업데이트"""
        # 최적 매칭 찾기
        best_match, match_score = self.find_best_match(current_pitch, current_time)
        
        if best_match and match_score >= self.sync_threshold:
            # 동기화 성공
            self.sync_state['current_position'] = best_match['index']
            self.sync_state['sync_confidence'] = match_score
            self.sync_state['is_synced'] = True
            self.sync_state['last_sync_time'] = current_time
            
            # 동기화 히스토리에 추가
            self.sync_state['sync_history'].append({
                'time': current_time,
                'position': best_match['index'],
                'confidence': match_score,
                'pitch': current_pitch,
                'note': best_match['pitch']
            })
        else:
            # 동기화 실패
            self.sync_state['sync_confidence'] = match_score
            self.sync_state['is_synced'] = False
    
    def get_current_note_info(self) -> Dict:
        """현재 연주 중인 음표 정보 반환"""
        if not self.sync_state['is_synced']:
            return {
                'note': None,
                'position': -1,
                'confidence': 0.0,
                'status': 'not_synced'
            }
        
        notes = self.sheet_music_data['notes']
        current_pos = self.sync_state['current_position']
        
        if 0 <= current_pos < len(notes):
            current_note = notes[current_pos]
            return {
                'note': current_note,
                'position': current_pos,
                'confidence': self.sync_state['sync_confidence'],
                'status': 'synced',
                'next_notes': notes[current_pos:current_pos+3]  # 다음 3개 음표
            }
        else:
            return {
                'note': None,
                'position': -1,
                'confidence': 0.0,
                'status': 'out_of_range'
            }
    
    def start_sync_monitoring(self, duration: float = 30.0):
        """동기화 모니터링 시작"""
        print(f"🔄 동기화 모니터링 시작 (최대 {duration}초)...")
        
        # 악보 데이터 로드
        self.load_sheet_music()
        
        # 오디오 분석
        print("🔍 오디오 분석 중...")
        audio_analysis = self.audio_processor.analyze_full_audio()
        
        # 실시간 동기화 시뮬레이션
        results = audio_analysis['detailed_results']
        sync_results = []
        
        for result in results[:int(duration * 10)]:  # 0.1초 단위로 제한
            current_time = result['time']
            current_pitch = result['pitch']
            
            # 동기화 상태 업데이트
            self.update_sync_state(current_pitch, current_time)
            
            # 결과 저장
            sync_info = self.get_current_note_info()
            sync_results.append({
                'time': current_time,
                'audio_pitch': current_pitch,
                'sync_info': sync_info,
                'sync_state': self.sync_state.copy()
            })
            
            # 진행 상황 출력
            if len(sync_results) % 50 == 0:  # 5초마다 출력
                print(f"⏱️ {current_time:.1f}초: "
                      f"음높이={current_pitch:.1f}Hz, "
                      f"위치={sync_info['position']}, "
                      f"신뢰도={sync_info['confidence']:.2f}")
        
        return sync_results
    
    def visualize_sync_results(self, sync_results: List[Dict]):
        """동기화 결과 시각화"""
        print("📊 동기화 결과 시각화...")
        
        times = [r['time'] for r in sync_results]
        audio_pitches = [r['audio_pitch'] for r in sync_results]
        sync_positions = [r['sync_info']['position'] for r in sync_results]
        sync_confidences = [r['sync_info']['confidence'] for r in sync_results]
        
        # 그래프 생성
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. 오디오 음높이 vs 악보 음높이
        axes[0].plot(times, audio_pitches, 'b-', alpha=0.7, label='Audio Pitch')
        
        # 악보 음높이 표시
        if self.sheet_music_data and 'notes' in self.sheet_music_data:
            notes = self.sheet_music_data['notes']
            note_times = [note['start_time'] for note in notes]
            note_freqs = [self.note_name_to_frequency(note['pitch']) for note in notes]
            axes[0].scatter(note_times, note_freqs, c='red', s=50, alpha=0.8, label='Sheet Music')
        
        axes[0].set_title('Audio Pitch vs Sheet Music')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 동기화 위치
        axes[1].plot(times, sync_positions, 'g-', alpha=0.7)
        axes[1].set_title('Sync Position')
        axes[1].set_ylabel('Note Index')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 동기화 신뢰도
        axes[2].plot(times, sync_confidences, 'r-', alpha=0.7)
        axes[2].axhline(y=self.sync_threshold, color='orange', linestyle='--', label='Sync Threshold')
        axes[2].set_title('Sync Confidence')
        axes[2].set_ylabel('Confidence')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sync_analysis_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 시각화 완료: sync_analysis_result.png")
    
    def save_sync_results(self, sync_results: List[Dict], filename: str = 'sync_results.json'):
        """동기화 결과 저장"""
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, deque):
                return list(obj)  # deque를 list로 변환
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy_types(sync_results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 동기화 결과 저장: {filename}")
    
    def get_sync_statistics(self, sync_results: List[Dict]) -> Dict:
        """동기화 통계 계산"""
        total_samples = len(sync_results)
        synced_samples = sum(1 for r in sync_results if r['sync_info']['status'] == 'synced')
        
        avg_confidence = np.mean([r['sync_info']['confidence'] for r in sync_results])
        max_confidence = max([r['sync_info']['confidence'] for r in sync_results])
        
        sync_rate = synced_samples / total_samples if total_samples > 0 else 0
        
        return {
            'total_samples': total_samples,
            'synced_samples': synced_samples,
            'sync_rate': sync_rate,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'sync_threshold': self.sync_threshold
        }

def main():
    """메인 함수"""
    print("🎼 악보-연주 싱크 시스템 시작")
    
    # 파일 경로
    sheet_music_path = "sheet music_1.png"
    audio_file_path = "주 품에 품으소서.mp3"
    
    try:
        # 싱크 시스템 초기화
        sync_system = SheetMusicSyncSystem(sheet_music_path, audio_file_path)
        
        # 동기화 모니터링 실행
        sync_results = sync_system.start_sync_monitoring(duration=30.0)
        
        # 통계 계산
        stats = sync_system.get_sync_statistics(sync_results)
        print(f"\n📊 동기화 통계:")
        print(f"   - 총 샘플 수: {stats['total_samples']}")
        print(f"   - 동기화된 샘플: {stats['synced_samples']}")
        print(f"   - 동기화율: {stats['sync_rate']:.2%}")
        print(f"   - 평균 신뢰도: {stats['avg_confidence']:.3f}")
        print(f"   - 최대 신뢰도: {stats['max_confidence']:.3f}")
        
        # 결과 시각화
        sync_system.visualize_sync_results(sync_results)
        
        # 결과 저장
        sync_system.save_sync_results(sync_results)
        
        print("\n✅ 악보-연주 싱크 시스템 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
