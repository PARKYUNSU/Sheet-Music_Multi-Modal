"""
고급 OMR 시스템 - 실시간 악보-연주 싱크를 위한 개선된 버전
"""

import cv2
import numpy as np
import librosa
import pretty_midi
import music21
from typing import List, Tuple, Dict, Optional
# import torch
# import torchvision.transforms as transforms
from PIL import Image
import json
import time

class AdvancedOMR:
    def __init__(self):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        
    def setup_models(self):
        """딥러닝 모델 초기화"""
        # YOLO 모델 로드 (음표 검출용)
        # self.note_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        # 음표 분류 모델 로드
        # self.note_classifier = self.load_note_classifier()
        
        print("모델 초기화 완료")
    
    def preprocess_image_advanced(self, image_path: str) -> np.ndarray:
        """고급 이미지 전처리"""
        # 고해상도 이미지 로드
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 해상도 향상 (Super Resolution)
        # image = self.enhance_resolution(image)
        
        # 색상 공간 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 적응적 임계값 처리
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def detect_staff_lines_advanced(self, binary_image: np.ndarray) -> List[Dict]:
        """고급 오선 검출"""
        height, width = binary_image.shape
        
        # Hough 변환으로 직선 검출
        lines = cv2.HoughLinesP(
            binary_image, 1, np.pi/180, threshold=100,
            minLineLength=width//2, maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # 수평선만 필터링
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # 수평선만
                horizontal_lines.append({'y': (y1 + y2) // 2, 'x1': x1, 'x2': x2})
        
        # 오선 그룹화 (5줄씩)
        staff_groups = self.group_staff_lines(horizontal_lines)
        
        return staff_groups
    
    def group_staff_lines(self, lines: List[Dict]) -> List[Dict]:
        """오선을 5줄씩 그룹화"""
        if not lines:
            return []
        
        # Y 좌표로 정렬
        lines.sort(key=lambda x: x['y'])
        
        staff_groups = []
        current_group = [lines[0]]
        
        for line in lines[1:]:
            # 이전 줄과의 거리가 20픽셀 이내면 같은 오선 그룹
            if line['y'] - current_group[-1]['y'] < 20:
                current_group.append(line)
            else:
                # 5줄이면 완성된 오선 그룹
                if len(current_group) >= 5:
                    staff_groups.append({
                        'lines': current_group[:5],
                        'top': current_group[0]['y'],
                        'bottom': current_group[4]['y']
                    })
                current_group = [line]
        
        # 마지막 그룹 처리
        if len(current_group) >= 5:
            staff_groups.append({
                'lines': current_group[:5],
                'top': current_group[0]['y'],
                'bottom': current_group[4]['y']
            })
        
        return staff_groups
    
    def detect_notes_advanced(self, binary_image: np.ndarray, staff_groups: List[Dict]) -> List[Dict]:
        """고급 음표 검출"""
        notes = []
        
        for staff in staff_groups:
            # 오선 영역 추출
            staff_region = binary_image[staff['top']-20:staff['bottom']+20, :]
            
            # 음표 후보 검출
            note_candidates = self.find_note_candidates(staff_region)
            
            # 각 후보를 분석하여 음표 정보 추출
            for candidate in note_candidates:
                note_info = self.analyze_note_candidate(candidate, staff)
                if note_info:
                    notes.append(note_info)
        
        return notes
    
    def find_note_candidates(self, staff_region: np.ndarray) -> List[Dict]:
        """음표 후보 검출"""
        # 컨투어 검출
        contours, _ = cv2.findContours(staff_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000:  # 적절한 크기의 객체만
                x, y, w, h = cv2.boundingRect(contour)
                candidates.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        return candidates
    
    def analyze_note_candidate(self, candidate: Dict, staff: Dict) -> Optional[Dict]:
        """음표 후보 분석"""
        x, y, w, h = candidate['bbox']
        
        # 음표 종류 분류 (간단한 휴리스틱)
        note_type = self.classify_note_type(candidate)
        
        # 음높이 계산
        pitch = self.calculate_pitch(y, staff)
        
        # 음표 길이 추정
        duration = self.estimate_duration(candidate)
        
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'type': note_type,
            'pitch': pitch,
            'duration': duration,
            'staff_index': staff.get('index', 0)
        }
    
    def classify_note_type(self, candidate: Dict) -> str:
        """음표 종류 분류"""
        # 간단한 휴리스틱 분류
        # 실제로는 딥러닝 모델 사용 권장
        area = candidate['area']
        x, y, w, h = candidate['bbox']
        
        if area < 100:
            return 'eighth'  # 8분음표
        elif area < 200:
            return 'quarter'  # 4분음표
        else:
            return 'half'  # 2분음표
    
    def calculate_pitch(self, y: int, staff: Dict) -> str:
        """음높이 계산"""
        # 오선과의 상대적 위치로 음높이 계산
        staff_lines = staff['lines']
        staff_height = staff_lines[4]['y'] - staff_lines[0]['y']
        
        # 오선 간격
        line_spacing = staff_height / 4
        
        # 음표가 어느 오선/간에 있는지 계산
        relative_position = (y - staff_lines[0]['y']) / line_spacing
        
        # 음높이 매핑 (간단한 예시)
        pitch_map = {
            0: 'E5', 0.5: 'F5', 1: 'G5', 1.5: 'A5',
            2: 'B5', 2.5: 'C6', 3: 'D6', 3.5: 'E6',
            4: 'F6'
        }
        
        # 가장 가까운 위치 찾기
        closest_pos = min(pitch_map.keys(), key=lambda x: abs(x - relative_position))
        return pitch_map[closest_pos]
    
    def estimate_duration(self, candidate: Dict) -> float:
        """음표 길이 추정"""
        # 음표 종류에 따른 길이
        note_type = self.classify_note_type(candidate)
        duration_map = {
            'whole': 4.0,
            'half': 2.0,
            'quarter': 1.0,
            'eighth': 0.5,
            'sixteenth': 0.25
        }
        return duration_map.get(note_type, 1.0)
    
    def extract_lyrics_advanced(self, image_path: str, staff_groups: List[Dict]) -> List[Dict]:
        """고급 가사 추출"""
        image = Image.open(image_path)
        
        lyrics = []
        for i, staff in enumerate(staff_groups):
            # 각 오선 아래 영역에서 가사 추출
            staff_bottom = staff['bottom']
            lyric_region = image.crop((0, staff_bottom, image.width, staff_bottom + 50))
            
            # OCR 적용
            # text = pytesseract.image_to_string(lyric_region, lang='kor')
            
            # 가사 정보 저장
            lyrics.append({
                'staff_index': i,
                'text': '',  # OCR 결과
                'y_position': staff_bottom + 25
            })
        
        return lyrics
    
    def create_midi_sequence(self, notes: List[Dict], tempo: int = 120) -> pretty_midi.PrettyMIDI:
        """MIDI 시퀀스 생성"""
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # 피아노
        
        current_time = 0.0
        
        for note in notes:
            # 음높이를 MIDI 번호로 변환
            midi_note = self.pitch_to_midi(note['pitch'])
            
            # 음표 길이 계산
            duration = note['duration'] * (60.0 / tempo)
            
            # MIDI 노트 생성
            midi_note_obj = pretty_midi.Note(
                velocity=80,
                pitch=midi_note,
                start=current_time,
                end=current_time + duration
            )
            
            piano.notes.append(midi_note_obj)
            current_time += duration
        
        midi.instruments.append(piano)
        return midi
    
    def pitch_to_midi(self, pitch: str) -> int:
        """음높이를 MIDI 번호로 변환"""
        # 간단한 매핑 (실제로는 더 정교한 변환 필요)
        pitch_map = {
            'C4': 60, 'D4': 62, 'E4': 64, 'F4': 65, 'G4': 67, 'A4': 69, 'B4': 71,
            'C5': 72, 'D5': 74, 'E5': 76, 'F5': 77, 'G5': 79, 'A5': 81, 'B5': 83,
            'C6': 84, 'D6': 86, 'E6': 88, 'F6': 89
        }
        return pitch_map.get(pitch, 60)
    
    def process_sheet_music(self, image_path: str) -> Dict:
        """악보 처리 메인 함수"""
        start_time = time.time()
        
        # 이미지 전처리
        binary_image = self.preprocess_image_advanced(image_path)
        
        # 오선 검출
        staff_groups = self.detect_staff_lines_advanced(binary_image)
        
        # 음표 검출
        notes = self.detect_notes_advanced(binary_image, staff_groups)
        
        # 가사 추출
        lyrics = self.extract_lyrics_advanced(image_path, staff_groups)
        
        # MIDI 생성
        midi = self.create_midi_sequence(notes)
        
        processing_time = time.time() - start_time
        
        return {
            'notes': notes,
            'lyrics': lyrics,
            'staff_groups': staff_groups,
            'midi': midi,
            'processing_time': processing_time,
            'total_notes': len(notes)
        }

# 사용 예시
if __name__ == "__main__":
    omr = AdvancedOMR()
    result = omr.process_sheet_music('sheet music_1.png')
    
    print(f"처리 시간: {result['processing_time']:.2f}초")
    print(f"검출된 음표 수: {result['total_notes']}")
    
    # 결과를 JSON으로 저장 (JSON 직렬화 가능하도록 변환)
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
    
    result_serializable = convert_numpy_types({
        'notes': result['notes'],
        'lyrics': result['lyrics'],
        'processing_time': result['processing_time']
    })
    
    with open('omr_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_serializable, f, ensure_ascii=False, indent=2)
