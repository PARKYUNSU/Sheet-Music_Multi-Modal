"""
고급 악보-연주 싱크 시스템
간주, 반복, 1절/2절 등을 고려한 실제 음악 연주 구조 분석
"""

import json
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from collections import deque
import librosa
from improved_omr import ImprovedOMR
from real_time_audio_processor import RealTimeAudioProcessor

class AdvancedSyncSystem:
    def __init__(self, sheet_music_path: str, audio_file_path: str):
        """
        고급 악보-연주 싱크 시스템 초기화
        
        Args:
            sheet_music_path: 악보 이미지 경로
            audio_file_path: 오디오 파일 경로
        """
        self.sheet_music_path = sheet_music_path
        self.audio_file_path = audio_file_path
        
        # OMR 시스템 초기화
        print("🎼 고급 OMR 시스템 초기화...")
        self.omr = ImprovedOMR()
        self.sheet_music_data = None
        
        # 오디오 처리 시스템 초기화
        print("🎵 오디오 처리 시스템 초기화...")
        self.audio_processor = RealTimeAudioProcessor(audio_file_path)
        
        # 음악 구조 분석
        self.music_structure = {
            'intro_duration': 0.0,  # 간주 길이
            'verse_duration': 0.0,  # 1절 길이
            'total_verses': 0,  # 총 절 수
            'repeat_sections': [],  # 반복 구간들
            'tempo_changes': [],  # 템포 변화
            'structure_timeline': []  # 전체 구조 타임라인
        }
        
        # 동기화 상태
        self.sync_state = {
            'current_section': 'unknown',  # 현재 구간 (intro, verse1, verse2, etc.)
            'current_position': 0,  # 현재 연주 위치 (음표 인덱스)
            'sync_confidence': 0.0,  # 동기화 신뢰도
            'is_synced': False,  # 동기화 상태
            'tempo': 120.0,  # 현재 템포
            'repeat_count': 0,  # 반복 횟수
            'sync_history': deque(maxlen=100)
        }
        
        # 매칭 설정
        self.pitch_tolerance = 100  # 음높이 허용 오차 (센트) - 더 관대하게
        self.tempo_tolerance = 0.3  # 템포 허용 오차 (30%)
        self.sync_threshold = 0.5  # 동기화 최소 신뢰도 - 낮춤
        
        print("✅ 고급 악보-연주 싱크 시스템 초기화 완료")
    
    def analyze_music_structure(self):
        """음악 구조 분석 (간주, 1절/2절, 반복 구간 등)"""
        print("🔍 음악 구조 분석 시작...")
        
        # 오디오 분석
        audio_analysis = self.audio_processor.analyze_full_audio()
        results = audio_analysis['detailed_results']
        
        # 1. 간주 구간 찾기 (처음 30초 내에서 음높이 변화가 적은 구간)
        intro_candidates = self.find_intro_section(results[:300])  # 처음 30초
        
        # 2. 1절/2절 패턴 찾기 (유사한 멜로디 패턴)
        verse_patterns = self.find_verse_patterns(results)
        
        # 3. 반복 구간 찾기
        repeat_sections = self.find_repeat_sections(results)
        
        # 4. 템포 변화 분석
        tempo_changes = self.analyze_tempo_changes(results)
        
        # 구조 정보 저장
        self.music_structure.update({
            'intro_duration': intro_candidates['duration'] if intro_candidates else 0.0,
            'verse_duration': verse_patterns['duration'] if verse_patterns else 0.0,
            'total_verses': verse_patterns['count'] if verse_patterns else 0,
            'repeat_sections': repeat_sections,
            'tempo_changes': tempo_changes
        })
        
        # 전체 구조 타임라인 생성
        self.create_structure_timeline()
        
        print(f"✅ 음악 구조 분석 완료:")
        print(f"   - 간주: {self.music_structure['intro_duration']:.1f}초")
        print(f"   - 1절 길이: {self.music_structure['verse_duration']:.1f}초")
        print(f"   - 총 절 수: {self.music_structure['total_verses']}")
        print(f"   - 반복 구간: {len(self.music_structure['repeat_sections'])}개")
    
    def find_intro_section(self, results: List[Dict]) -> Optional[Dict]:
        """간주 구간 찾기"""
        if len(results) < 50:
            return None
        
        # 음높이 변화가 적고 볼륨이 낮은 구간 찾기
        pitch_variance = []
        volume_avg = []
        
        window_size = 20  # 2초 윈도우
        
        for i in range(0, len(results) - window_size, 10):
            window = results[i:i+window_size]
            pitches = [r['pitch'] for r in window if r['pitch'] > 0]
            volumes = [r['volume'] for r in window]
            
            if len(pitches) > 5:
                pitch_var = np.var(pitches)
                vol_avg = np.mean(volumes)
                pitch_variance.append(pitch_var)
                volume_avg.append(vol_avg)
            else:
                pitch_variance.append(float('inf'))
                volume_avg.append(1.0)
        
        # 간주 후보: 음높이 변화가 적고 볼륨이 낮은 구간
        if pitch_variance:
            min_var_idx = np.argmin(pitch_variance)
            if pitch_variance[min_var_idx] < 1000 and volume_avg[min_var_idx] < 0.1:
                start_time = results[min_var_idx * 10]['time']
                end_time = results[min_var_idx * 10 + window_size]['time']
                return {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'pitch_variance': pitch_variance[min_var_idx],
                    'volume_avg': volume_avg[min_var_idx]
                }
        
        return None
    
    def find_verse_patterns(self, results: List[Dict]) -> Optional[Dict]:
        """1절/2절 패턴 찾기"""
        # 간단한 패턴 매칭: 유사한 음높이 시퀀스 찾기
        pitch_sequence = [r['pitch'] for r in results if r['pitch'] > 0]
        
        if len(pitch_sequence) < 100:
            return None
        
        # 슬라이딩 윈도우로 패턴 찾기
        window_size = 50  # 5초 윈도우
        similarities = []
        
        for i in range(0, len(pitch_sequence) - window_size * 2, 10):
            pattern1 = pitch_sequence[i:i+window_size]
            
            for j in range(i + window_size, len(pitch_sequence) - window_size, 10):
                pattern2 = pitch_sequence[j:j+window_size]
                
                # 패턴 유사도 계산 (간단한 상관계수)
                if len(pattern1) == len(pattern2):
                    similarity = np.corrcoef(pattern1, pattern2)[0, 1]
                    if not np.isnan(similarity):
                        similarities.append({
                            'start1': i,
                            'start2': j,
                            'similarity': similarity,
                            'time1': results[i]['time'],
                            'time2': results[j]['time']
                        })
        
        # 높은 유사도를 가진 패턴들 찾기
        if similarities:
            high_similarities = [s for s in similarities if s['similarity'] > 0.7]
            if high_similarities:
                best_match = max(high_similarities, key=lambda x: x['similarity'])
                verse_duration = best_match['time2'] - best_match['time1']
                
                # 총 절 수 추정
                total_duration = results[-1]['time']
                estimated_verses = int(total_duration / verse_duration)
                
                return {
                    'duration': verse_duration,
                    'count': estimated_verses,
                    'similarity': best_match['similarity'],
                    'pattern_match': best_match
                }
        
        return None
    
    def find_repeat_sections(self, results: List[Dict]) -> List[Dict]:
        """반복 구간 찾기"""
        repeat_sections = []
        
        # 간단한 반복 구간 검출: 동일한 음높이 패턴이 반복되는 구간
        pitch_sequence = [r['pitch'] for r in results if r['pitch'] > 0]
        
        if len(pitch_sequence) < 200:
            return repeat_sections
        
        # 10초 단위로 패턴 검사
        pattern_size = 100  # 10초 패턴
        
        for i in range(0, len(pitch_sequence) - pattern_size * 2, 50):
            pattern = pitch_sequence[i:i+pattern_size]
            
            # 같은 패턴이 반복되는지 확인
            for j in range(i + pattern_size, len(pitch_sequence) - pattern_size, 50):
                candidate = pitch_sequence[j:j+pattern_size]
                
                if len(pattern) == len(candidate):
                    similarity = np.corrcoef(pattern, candidate)[0, 1]
                    if not np.isnan(similarity) and similarity > 0.8:
                        repeat_sections.append({
                            'start_time': results[i]['time'],
                            'end_time': results[i + pattern_size]['time'],
                            'repeat_start': results[j]['time'],
                            'repeat_end': results[j + pattern_size]['time'],
                            'similarity': similarity
                        })
        
        return repeat_sections
    
    def analyze_tempo_changes(self, results: List[Dict]) -> List[Dict]:
        """템포 변화 분석"""
        tempo_changes = []
        
        # 볼륨과 음높이 변화를 기반으로 템포 변화 추정
        window_size = 100  # 10초 윈도우
        
        for i in range(0, len(results) - window_size, 50):
            window = results[i:i+window_size]
            
            # 볼륨 변화율 계산
            volumes = [r['volume'] for r in window]
            volume_changes = np.diff(volumes)
            volume_variance = np.var(volume_changes)
            
            # 음높이 변화율 계산
            pitches = [r['pitch'] for r in window if r['pitch'] > 0]
            if len(pitches) > 10:
                pitch_changes = np.diff(pitches)
                pitch_variance = np.var(pitch_changes)
                
                # 템포 변화 추정 (볼륨과 음높이 변화 기반)
                estimated_tempo = 120 + (volume_variance * 10) + (pitch_variance * 0.1)
                estimated_tempo = max(60, min(180, estimated_tempo))  # 60-180 BPM 범위
                
                tempo_changes.append({
                    'time': results[i]['time'],
                    'tempo': estimated_tempo,
                    'volume_variance': volume_variance,
                    'pitch_variance': pitch_variance
                })
        
        return tempo_changes
    
    def create_structure_timeline(self):
        """전체 구조 타임라인 생성"""
        timeline = []
        current_time = 0.0
        
        # 간주
        if self.music_structure['intro_duration'] > 0:
            timeline.append({
                'type': 'intro',
                'start_time': current_time,
                'end_time': current_time + self.music_structure['intro_duration'],
                'description': '간주'
            })
            current_time += self.music_structure['intro_duration']
        
        # 1절, 2절 등
        verse_duration = self.music_structure['verse_duration']
        total_verses = self.music_structure['total_verses']
        
        for i in range(total_verses):
            timeline.append({
                'type': f'verse{i+1}',
                'start_time': current_time,
                'end_time': current_time + verse_duration,
                'description': f'{i+1}절'
            })
            current_time += verse_duration
        
        # 반복 구간들
        for i, repeat in enumerate(self.music_structure['repeat_sections']):
            timeline.append({
                'type': f'repeat{i+1}',
                'start_time': repeat['start_time'],
                'end_time': repeat['end_time'],
                'description': f'반복 구간 {i+1}'
            })
        
        self.music_structure['structure_timeline'] = timeline
    
    def load_sheet_music(self):
        """악보 데이터 로드"""
        print("📄 악보 데이터 로딩...")
        
        # OMR로 악보 분석
        self.sheet_music_data = self.omr.process_sheet_music_improved(self.sheet_music_path)
        
        # 음표 데이터를 시간축으로 변환
        self.convert_notes_to_timeline()
        
        print(f"✅ 악보 로드 완료: {len(self.sheet_music_data['notes'])}개 음표")
    
    def convert_notes_to_timeline(self):
        """음표 데이터를 시간축으로 변환 (반복 구조 고려)"""
        if not self.sheet_music_data or 'notes' not in self.sheet_music_data:
            return
        
        notes = self.sheet_music_data['notes']
        tempo = self.sync_state['tempo']
        
        # 기본 1절 길이 계산
        base_duration = 0.0
        for note in notes:
            note_duration = note.get('duration', 1.0)
            base_duration += (note_duration * 60.0 / tempo)
        
        # 음악 구조에 맞게 시간축 조정
        current_time = 0.0
        
        # 간주 시간 추가
        if self.music_structure['intro_duration'] > 0:
            current_time += self.music_structure['intro_duration']
        
        # 각 음표에 시간 정보 추가
        for i, note in enumerate(notes):
            note['start_time'] = current_time
            
            note_duration = note.get('duration', 1.0)
            note['end_time'] = current_time + (note_duration * 60.0 / tempo)
            
            current_time = note['end_time']
            note['index'] = i
        
        print(f"✅ 시간축 변환 완료: 총 {current_time:.2f}초")
    
    def find_best_match_with_structure(self, current_pitch: float, current_time: float) -> Tuple[Optional[Dict], float]:
        """음악 구조를 고려한 최적 매칭"""
        if not self.sheet_music_data or 'notes' not in self.sheet_music_data:
            return None, 0.0
        
        notes = self.sheet_music_data['notes']
        best_match = None
        best_score = 0.0
        
        # 현재 시간이 어느 구간에 속하는지 확인
        current_section = self.get_current_section(current_time)
        
        # 해당 구간에 맞는 음표들 검사
        for note in notes:
            note_freq = self.note_name_to_frequency(note['pitch'])
            
            # 시간 매칭 (구간별로 다른 윈도우 적용)
            time_window = self.get_time_window_for_section(current_section)
            time_diff = abs(current_time - note['start_time'])
            
            if time_diff > time_window:
                continue
            
            time_score = 1.0 - (time_diff / time_window)
            
            # 음높이 매칭
            pitch_score = self.match_pitch(current_pitch, note_freq)
            
            # 구간별 가중치 적용
            section_weight = self.get_section_weight(current_section)
            
            # 종합 점수
            total_score = (0.4 * time_score + 0.6 * pitch_score) * section_weight
            
            if total_score > best_score:
                best_score = total_score
                best_match = note
        
        return best_match, best_score
    
    def get_current_section(self, current_time: float) -> str:
        """현재 시간이 속하는 구간 반환"""
        for section in self.music_structure['structure_timeline']:
            if section['start_time'] <= current_time <= section['end_time']:
                return section['type']
        return 'unknown'
    
    def get_time_window_for_section(self, section: str) -> float:
        """구간별 시간 윈도우 반환"""
        windows = {
            'intro': 10.0,  # 간주는 더 넓은 윈도우
            'verse1': 5.0,
            'verse2': 5.0,
            'repeat1': 8.0,  # 반복 구간은 중간 윈도우
            'unknown': 15.0  # 알 수 없는 구간은 넓은 윈도우
        }
        return windows.get(section, 5.0)
    
    def get_section_weight(self, section: str) -> float:
        """구간별 가중치 반환"""
        weights = {
            'intro': 0.8,  # 간주는 낮은 가중치
            'verse1': 1.0,  # 1절은 높은 가중치
            'verse2': 1.0,  # 2절도 높은 가중치
            'repeat1': 0.9,  # 반복 구간은 중간 가중치
            'unknown': 0.5  # 알 수 없는 구간은 낮은 가중치
        }
        return weights.get(section, 1.0)
    
    def match_pitch(self, current_pitch: float, note_frequency: float) -> float:
        """음높이 매칭 점수 계산"""
        if current_pitch <= 0 or note_frequency <= 0:
            return 0.0
        
        cents_diff = abs(self.frequency_to_cents(current_pitch, note_frequency))
        
        if cents_diff <= self.pitch_tolerance:
            return 1.0 - (cents_diff / self.pitch_tolerance)
        else:
            return 0.0
    
    def frequency_to_cents(self, freq1: float, freq2: float) -> float:
        """두 주파수 간의 센트 차이 계산"""
        if freq1 <= 0 or freq2 <= 0:
            return float('inf')
        return 1200 * np.log2(freq2 / freq1)
    
    def note_name_to_frequency(self, note_name: str) -> float:
        """음표명을 주파수로 변환"""
        if note_name == "Silence":
            return 0.0
        
        note_map = {
            'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
            'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        }
        
        try:
            note = note_name[:-1]
            octave = int(note_name[-1])
            semitones = note_map[note] + (octave - 4) * 12
            frequency = 440.0 * (2 ** (semitones / 12))
            return frequency
        except:
            return 0.0
    
    def start_advanced_sync_monitoring(self, duration: float = 60.0):
        """고급 동기화 모니터링 시작"""
        print(f"🔄 고급 동기화 모니터링 시작 (최대 {duration}초)...")
        
        # 음악 구조 분석
        self.analyze_music_structure()
        
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
            
            # 고급 매칭 수행
            best_match, match_score = self.find_best_match_with_structure(current_pitch, current_time)
            
            # 동기화 상태 업데이트
            if best_match and match_score >= self.sync_threshold:
                self.sync_state['current_position'] = best_match['index']
                self.sync_state['sync_confidence'] = match_score
                self.sync_state['is_synced'] = True
                self.sync_state['current_section'] = self.get_current_section(current_time)
            else:
                self.sync_state['sync_confidence'] = match_score
                self.sync_state['is_synced'] = False
            
            # 결과 저장
            sync_results.append({
                'time': current_time,
                'audio_pitch': current_pitch,
                'matched_note': best_match,
                'match_score': match_score,
                'current_section': self.sync_state['current_section'],
                'sync_state': self.sync_state.copy()
            })
            
            # 진행 상황 출력
            if len(sync_results) % 100 == 0:  # 10초마다 출력
                section = self.sync_state['current_section']
                print(f"⏱️ {current_time:.1f}초 [{section}]: "
                      f"음높이={current_pitch:.1f}Hz, "
                      f"매칭점수={match_score:.3f}, "
                      f"동기화={'✅' if self.sync_state['is_synced'] else '❌'}")
        
        return sync_results
    
    def visualize_advanced_sync_results(self, sync_results: List[Dict]):
        """고급 동기화 결과 시각화"""
        print("📊 고급 동기화 결과 시각화...")
        
        times = [r['time'] for r in sync_results]
        audio_pitches = [r['audio_pitch'] for r in sync_results]
        match_scores = [r['match_score'] for r in sync_results]
        sections = [r['current_section'] for r in sync_results]
        
        # 그래프 생성
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 1. 오디오 음높이
        axes[0].plot(times, audio_pitches, 'b-', alpha=0.7, label='Audio Pitch')
        axes[0].set_title('Audio Pitch Over Time')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 매칭 점수
        axes[1].plot(times, match_scores, 'g-', alpha=0.7)
        axes[1].axhline(y=self.sync_threshold, color='orange', linestyle='--', label='Sync Threshold')
        axes[1].set_title('Matching Score')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 음악 구간
        section_colors = {'intro': 'red', 'verse1': 'blue', 'verse2': 'green', 'repeat1': 'orange', 'unknown': 'gray'}
        for i, section in enumerate(sections):
            if section in section_colors:
                axes[2].scatter(times[i], 1, c=section_colors[section], s=10, alpha=0.7)
        axes[2].set_title('Music Sections')
        axes[2].set_ylabel('Section')
        axes[2].set_ylim(0.5, 1.5)
        axes[2].grid(True, alpha=0.3)
        
        # 4. 동기화 상태
        sync_status = [1 if r['sync_state']['is_synced'] else 0 for r in sync_results]
        axes[3].plot(times, sync_status, 'r-', alpha=0.7)
        axes[3].set_title('Sync Status')
        axes[3].set_ylabel('Synced (1/0)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_sync_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 고급 시각화 완료: advanced_sync_analysis.png")
    
    def save_advanced_sync_results(self, sync_results: List[Dict], filename: str = 'advanced_sync_results.json'):
        """고급 동기화 결과 저장"""
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, deque):
                return list(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # 결과와 음악 구조 정보 모두 저장
        save_data = {
            'sync_results': sync_results,
            'music_structure': self.music_structure,
            'sync_statistics': self.get_advanced_sync_statistics(sync_results)
        }
        
        serializable_data = convert_numpy_types(save_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 고급 동기화 결과 저장: {filename}")
    
    def get_advanced_sync_statistics(self, sync_results: List[Dict]) -> Dict:
        """고급 동기화 통계 계산"""
        total_samples = len(sync_results)
        synced_samples = sum(1 for r in sync_results if r['sync_state']['is_synced'])
        
        avg_score = np.mean([r['match_score'] for r in sync_results])
        max_score = max([r['match_score'] for r in sync_results])
        
        # 구간별 통계
        section_stats = {}
        for section in set(r['current_section'] for r in sync_results):
            section_results = [r for r in sync_results if r['current_section'] == section]
            section_synced = sum(1 for r in section_results if r['sync_state']['is_synced'])
            section_stats[section] = {
                'total_samples': len(section_results),
                'synced_samples': section_synced,
                'sync_rate': section_synced / len(section_results) if section_results else 0,
                'avg_score': np.mean([r['match_score'] for r in section_results])
            }
        
        return {
            'total_samples': total_samples,
            'synced_samples': synced_samples,
            'overall_sync_rate': synced_samples / total_samples if total_samples > 0 else 0,
            'avg_match_score': avg_score,
            'max_match_score': max_score,
            'section_statistics': section_stats,
            'music_structure_summary': {
                'intro_duration': self.music_structure['intro_duration'],
                'verse_duration': self.music_structure['verse_duration'],
                'total_verses': self.music_structure['total_verses'],
                'repeat_sections_count': len(self.music_structure['repeat_sections'])
            }
        }

def main():
    """메인 함수"""
    print("🎼 고급 악보-연주 싱크 시스템 시작")
    
    # 파일 경로
    sheet_music_path = "sheet music_1.png"
    audio_file_path = "주 품에 품으소서.mp3"
    
    try:
        # 고급 싱크 시스템 초기화
        sync_system = AdvancedSyncSystem(sheet_music_path, audio_file_path)
        
        # 고급 동기화 모니터링 실행
        sync_results = sync_system.start_advanced_sync_monitoring(duration=60.0)
        
        # 통계 계산 및 출력
        stats = sync_system.get_advanced_sync_statistics(sync_results)
        print(f"\n📊 고급 동기화 통계:")
        print(f"   - 총 샘플 수: {stats['total_samples']}")
        print(f"   - 동기화된 샘플: {stats['synced_samples']}")
        print(f"   - 전체 동기화율: {stats['overall_sync_rate']:.2%}")
        print(f"   - 평균 매칭 점수: {stats['avg_match_score']:.3f}")
        print(f"   - 최대 매칭 점수: {stats['max_match_score']:.3f}")
        
        print(f"\n🎵 음악 구조 요약:")
        print(f"   - 간주: {stats['music_structure_summary']['intro_duration']:.1f}초")
        print(f"   - 1절 길이: {stats['music_structure_summary']['verse_duration']:.1f}초")
        print(f"   - 총 절 수: {stats['music_structure_summary']['total_verses']}")
        print(f"   - 반복 구간: {stats['music_structure_summary']['repeat_sections_count']}개")
        
        print(f"\n📈 구간별 동기화율:")
        for section, section_stats in stats['section_statistics'].items():
            print(f"   - {section}: {section_stats['sync_rate']:.2%} ({section_stats['synced_samples']}/{section_stats['total_samples']})")
        
        # 결과 시각화
        sync_system.visualize_advanced_sync_results(sync_results)
        
        # 결과 저장
        sync_system.save_advanced_sync_results(sync_results)
        
        print("\n✅ 고급 악보-연주 싱크 시스템 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
