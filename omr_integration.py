"""
OMR 통합 모듈 - 다양한 OMR 솔루션을 통합하여 사용
"""

import subprocess
import os
import json
import music21
import pretty_midi
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

class OMRIntegration:
    def __init__(self):
        self.setup_environment()
        
    def setup_environment(self):
        """환경 설정"""
        # Tesseract 경로 설정
        self.tesseract_path = '/opt/homebrew/bin/tesseract'
        
        # MuseScore 경로 (설치되어 있다면)
        self.musescore_path = self.find_musescore()
        
        print(f"MuseScore 경로: {self.musescore_path}")
        
    def find_musescore(self) -> Optional[str]:
        """MuseScore 설치 경로 찾기"""
        # MuseScore는 현재 설치되어 있지 않음
        return None
    
    def method1_musescore_omr(self, image_path: str, output_path: str) -> bool:
        """방법 1: MuseScore OMR 사용"""
        if not self.musescore_path:
            print("MuseScore가 설치되어 있지 않습니다.")
            return False
        
        try:
            # MuseScore OMR 명령어 실행
            cmd = [
                self.musescore_path,
                '--export-to', output_path,
                '--import-image', image_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"MuseScore OMR 성공: {output_path}")
                return True
            else:
                print(f"MuseScore OMR 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("MuseScore OMR 타임아웃")
            return False
        except Exception as e:
            print(f"MuseScore OMR 오류: {e}")
            return False
    
    def method2_advanced_omr(self, image_path: str) -> Dict:
        """방법 2: 고급 OMR (이전에 만든 advanced_omr.py 사용)"""
        try:
            # advanced_omr.py 파일이 같은 디렉토리에 있는지 확인
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            advanced_omr_path = os.path.join(current_dir, 'advanced_omr.py')
            
            if os.path.exists(advanced_omr_path):
                sys.path.insert(0, current_dir)
                from advanced_omr import AdvancedOMR
                
                omr = AdvancedOMR()
                result = omr.process_sheet_music(image_path)
                
                return {
                    'success': True,
                    'notes': result['notes'],
                    'lyrics': result['lyrics'],
                    'processing_time': result['processing_time']
                }
            else:
                print("advanced_omr.py 파일을 찾을 수 없습니다.")
                return {'success': False, 'error': 'advanced_omr.py file not found'}
            
        except ImportError as e:
            print(f"advanced_omr 모듈 import 오류: {e}")
            return {'success': False, 'error': f'Import error: {e}'}
        except Exception as e:
            print(f"고급 OMR 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def method3_music21_analysis(self, musicxml_path: str) -> Dict:
        """방법 3: music21을 사용한 음악 분석"""
        try:
            # MusicXML 파일 로드
            score = music21.converter.parse(musicxml_path)
            
            # 음표 정보 추출
            notes = []
            lyrics = []
            
            for part in score.parts:
                for element in part.flat:
                    if isinstance(element, music21.note.Note):
                        notes.append({
                            'pitch': str(element.pitch),
                            'quarterLength': element.quarterLength,
                            'offset': element.offset,
                            'measure': element.measureNumber if hasattr(element, 'measureNumber') else None
                        })
                    elif isinstance(element, music21.note.Rest):
                        notes.append({
                            'pitch': 'REST',
                            'quarterLength': element.quarterLength,
                            'offset': element.offset,
                            'measure': element.measureNumber if hasattr(element, 'measureNumber') else None
                        })
                    elif isinstance(element, music21.note.Chord):
                        chord_notes = [str(n.pitch) for n in element.notes]
                        notes.append({
                            'pitch': chord_notes,
                            'quarterLength': element.quarterLength,
                            'offset': element.offset,
                            'measure': element.measureNumber if hasattr(element, 'measureNumber') else None
                        })
            
            # 가사 추출
            for part in score.parts:
                for element in part.flat:
                    if isinstance(element, music21.note.Note) and element.lyric:
                        lyrics.append({
                            'text': element.lyric,
                            'offset': element.offset
                        })
            
            return {
                'success': True,
                'notes': notes,
                'lyrics': lyrics,
                'total_notes': len(notes),
                'tempo': score.metronomeMarkBoundaries()[0][2].number if score.metronomeMarkBoundaries() else 120
            }
            
        except Exception as e:
            print(f"music21 분석 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def method4_hybrid_approach(self, image_path: str) -> Dict:
        """방법 4: 하이브리드 접근법 (여러 방법 조합)"""
        results = {}
        
        # 1. 고급 OMR 시도
        print("1. 고급 OMR 시도 중...")
        omr_result = self.method2_advanced_omr(image_path)
        results['advanced_omr'] = omr_result
        
        # 2. MuseScore OMR은 현재 사용하지 않음
        print("2. MuseScore OMR은 현재 사용하지 않습니다.")
        
        # 결과 통합
        return self.merge_results(results)
    
    def merge_results(self, results: Dict) -> Dict:
        """여러 OMR 결과를 통합"""
        merged = {
            'success': False,
            'notes': [],
            'lyrics': [],
            'confidence': 0.0,
            'method_used': 'none'
        }
        
        # 가장 좋은 결과 선택 (현재는 advanced_omr만 사용)
        if results.get('advanced_omr', {}).get('success'):
            merged.update(results['advanced_omr'])
            merged['method_used'] = 'advanced_omr'
            merged['confidence'] = 0.7
        
        return merged
    
    def create_midi_from_notes(self, notes: List[Dict], tempo: int = 120) -> pretty_midi.PrettyMIDI:
        """음표 정보로부터 MIDI 생성"""
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        current_time = 0.0
        
        for note in notes:
            if note['pitch'] == 'REST':
                current_time += note['quarterLength'] * (60.0 / tempo)
                continue
            
            # 음높이를 MIDI 번호로 변환
            if isinstance(note['pitch'], list):  # 화음
                for pitch in note['pitch']:
                    midi_note = self.pitch_to_midi(pitch)
                    duration = note['quarterLength'] * (60.0 / tempo)
                    
                    midi_note_obj = pretty_midi.Note(
                        velocity=80,
                        pitch=midi_note,
                        start=current_time,
                        end=current_time + duration
                    )
                    piano.notes.append(midi_note_obj)
            else:  # 단일 음표
                midi_note = self.pitch_to_midi(note['pitch'])
                duration = note['quarterLength'] * (60.0 / tempo)
                
                midi_note_obj = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_note,
                    start=current_time,
                    end=current_time + duration
                )
                piano.notes.append(midi_note_obj)
            
            current_time += note['quarterLength'] * (60.0 / tempo)
        
        midi.instruments.append(piano)
        return midi
    
    def pitch_to_midi(self, pitch: str) -> int:
        """음높이를 MIDI 번호로 변환"""
        # music21 pitch 형식 처리
        try:
            from music21 import pitch
            p = pitch.Pitch(pitch)
            return p.midi
        except:
            # 간단한 매핑
            pitch_map = {
                'C4': 60, 'D4': 62, 'E4': 64, 'F4': 65, 'G4': 67, 'A4': 69, 'B4': 71,
                'C5': 72, 'D5': 74, 'E5': 76, 'F5': 77, 'G5': 79, 'A5': 81, 'B5': 83,
                'C6': 84, 'D6': 86, 'E6': 88, 'F6': 89
            }
            return pitch_map.get(pitch, 60)
    
    def process_sheet_music(self, image_path: str, method: str = 'hybrid') -> Dict:
        """악보 처리 메인 함수"""
        print(f"악보 처리 시작: {image_path}")
        print(f"사용 방법: {method}")
        
        if method == 'musescore':
            musicxml_path = image_path.replace('.png', '_musescore.musicxml')
            success = self.method1_musescore_omr(image_path, musicxml_path)
            if success:
                return self.method3_music21_analysis(musicxml_path)
            else:
                return {'success': False, 'error': 'MuseScore OMR failed'}
        
        elif method == 'advanced':
            return self.method2_advanced_omr(image_path)
        
        elif method == 'hybrid':
            return self.method4_hybrid_approach(image_path)
        
        else:
            return {'success': False, 'error': f'Unknown method: {method}'}

# 사용 예시
if __name__ == "__main__":
    omr = OMRIntegration()
    
    # 악보 처리
    result = omr.process_sheet_music('sheet music_1.png', method='hybrid')
    
    if result['success']:
        print(f"처리 성공!")
        print(f"사용된 방법: {result.get('method_used', 'unknown')}")
        print(f"신뢰도: {result.get('confidence', 0.0):.2f}")
        print(f"음표 수: {result.get('total_notes', len(result.get('notes', [])))}")
        
        # MIDI 생성
        if result.get('notes'):
            midi = omr.create_midi_from_notes(result['notes'])
            midi.write('output.mid')
            print("MIDI 파일 생성: output.mid")
        
        # 결과 저장 (JSON 직렬화 가능하도록 변환)
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
        
        result_serializable = convert_numpy_types(result)
        
        with open('omr_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_serializable, f, ensure_ascii=False, indent=2)
        print("결과 저장: omr_result.json")
        
    else:
        print(f"처리 실패: {result.get('error', 'Unknown error')}")
