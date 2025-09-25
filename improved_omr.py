"""
개선된 OMR 시스템 - 음표 검출 알고리즘 개선
"""

import cv2
import numpy as np
import json
from typing import List, Dict, Optional
from advanced_omr import AdvancedOMR

class ImprovedOMR(AdvancedOMR):
    def __init__(self):
        super().__init__()
        print("개선된 OMR 시스템 초기화 완료")
    
    def preprocess_image_improved(self, image_path: str) -> np.ndarray:
        """개선된 이미지 전처리 - 오선 제거 및 음표 강화"""
        # 원본 이미지 로드
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 오선 검출 및 제거
        binary_without_staff = self.remove_staff_lines(gray)
        
        # 2. 음표 강화
        enhanced_notes = self.enhance_notes(binary_without_staff)
        
        return enhanced_notes
    
    def remove_staff_lines(self, gray_image: np.ndarray) -> np.ndarray:
        """오선 제거"""
        # 적응적 임계값으로 이진화
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 수평선 검출을 위한 구조 요소
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        
        # 수평선 검출
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 오선 제거 (수평선을 흰색으로 덮음)
        result = binary.copy()
        result[horizontal_lines > 0] = 0  # 오선을 검은색으로 변경
        
        return result
    
    def enhance_notes(self, binary_image: np.ndarray) -> np.ndarray:
        """음표 강화"""
        # 작은 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        
        # 음표 연결 (점들을 연결)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def find_note_candidates_improved(self, staff_region: np.ndarray) -> List[Dict]:
        """개선된 음표 후보 검출"""
        # 컨투어 검출
        contours, _ = cv2.findContours(staff_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 더 관대한 면적 조건
            if 5 < area < 5000:  # 기존 50-1000에서 5-5000으로 확장
                x, y, w, h = cv2.boundingRect(contour)
                
                # 종횡비 필터 (너무 길쭉한 것 제외)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # 적절한 종횡비
                    candidates.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        return candidates
    
    def classify_note_type_improved(self, candidate: Dict) -> str:
        """개선된 음표 종류 분류"""
        area = candidate['area']
        aspect_ratio = candidate['aspect_ratio']
        x, y, w, h = candidate['bbox']
        
        # 면적과 종횡비를 기반으로 분류
        if area < 20:
            return 'eighth'  # 8분음표
        elif area < 50:
            return 'quarter'  # 4분음표
        elif area < 100:
            return 'half'  # 2분음표
        else:
            return 'whole'  # 온음표
    
    def detect_notes_improved(self, binary_image: np.ndarray, staff_groups: List[Dict]) -> List[Dict]:
        """개선된 음표 검출"""
        notes = []
        
        for staff in staff_groups:
            # 오선 영역 추출 (여유 공간 확대)
            staff_region = binary_image[staff['top']-30:staff['bottom']+30, :]
            
            # 개선된 음표 후보 검출
            note_candidates = self.find_note_candidates_improved(staff_region)
            
            print(f"오선 그룹에서 {len(note_candidates)}개 후보 검출")
            
            # 각 후보를 분석하여 음표 정보 추출
            for candidate in note_candidates:
                note_info = self.analyze_note_candidate_improved(candidate, staff)
                if note_info:
                    notes.append(note_info)
        
        return notes
    
    def analyze_note_candidate_improved(self, candidate: Dict, staff: Dict) -> Optional[Dict]:
        """개선된 음표 후보 분석"""
        x, y, w, h = candidate['bbox']
        
        # 개선된 음표 종류 분류
        note_type = self.classify_note_type_improved(candidate)
        
        # 음높이 계산 (오선과의 상대적 위치)
        pitch = self.calculate_pitch_improved(y, staff)
        
        # 음표 길이 추정
        duration = self.estimate_duration_improved(candidate)
        
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'type': note_type,
            'pitch': pitch,
            'duration': duration,
            'staff_index': staff.get('index', 0),
            'area': candidate['area'],
            'aspect_ratio': candidate['aspect_ratio']
        }
    
    def calculate_pitch_improved(self, y: int, staff: Dict) -> str:
        """개선된 음높이 계산"""
        staff_lines = staff['lines']
        staff_height = staff_lines[4]['y'] - staff_lines[0]['y']
        
        # 오선 간격
        line_spacing = staff_height / 4
        
        # 음표가 어느 오선/간에 있는지 계산
        relative_position = (y - staff_lines[0]['y']) / line_spacing
        
        # 음높이 매핑 (더 정확한 계산)
        pitch_map = {
            -2: 'C4', -1.5: 'D4', -1: 'E4', -0.5: 'F4',
            0: 'G4', 0.5: 'A4', 1: 'B4', 1.5: 'C5',
            2: 'D5', 2.5: 'E5', 3: 'F5', 3.5: 'G5',
            4: 'A5', 4.5: 'B5', 5: 'C6', 5.5: 'D6',
            6: 'E6', 6.5: 'F6', 7: 'G6'
        }
        
        # 가장 가까운 위치 찾기
        closest_pos = min(pitch_map.keys(), key=lambda x: abs(x - relative_position))
        return pitch_map[closest_pos]
    
    def estimate_duration_improved(self, candidate: Dict) -> float:
        """개선된 음표 길이 추정"""
        note_type = self.classify_note_type_improved(candidate)
        duration_map = {
            'whole': 4.0,
            'half': 2.0,
            'quarter': 1.0,
            'eighth': 0.5,
            'sixteenth': 0.25
        }
        return duration_map.get(note_type, 1.0)
    
    def process_sheet_music_improved(self, image_path: str) -> Dict:
        """개선된 악보 처리 메인 함수"""
        import time
        start_time = time.time()
        
        # 원본 이미지로 오선 검출 (오선 제거 전에 해야 함)
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        binary_original = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 오선 검출 (원본 이진화 이미지에서)
        staff_groups = self.detect_staff_lines_advanced(binary_original)
        
        # 개선된 이미지 전처리 (오선 제거)
        binary_image = self.preprocess_image_improved(image_path)
        
        # 개선된 음표 검출
        notes = self.detect_notes_improved(binary_image, staff_groups)
        
        # 가사 추출
        lyrics = self.extract_lyrics_advanced(image_path, staff_groups)
        
        processing_time = time.time() - start_time
        
        return {
            'notes': notes,
            'lyrics': lyrics,
            'staff_groups': staff_groups,
            'processing_time': processing_time,
            'total_notes': len(notes)
        }

# 테스트 함수
def test_improved_omr():
    """개선된 OMR 시스템 테스트"""
    print("🚀 개선된 OMR 시스템 테스트 시작...")
    
    omr = ImprovedOMR()
    result = omr.process_sheet_music_improved('sheet music_1.png')
    
    print(f"\n📊 결과:")
    print(f"  - 처리 시간: {result['processing_time']:.3f}초")
    print(f"  - 검출된 음표 수: {result['total_notes']}")
    print(f"  - 오선 그룹 수: {len(result['staff_groups'])}")
    
    if result['notes']:
        print(f"\n🎵 검출된 음표들:")
        for i, note in enumerate(result['notes'][:10]):  # 처음 10개만 출력
            print(f"  {i+1}. {note['type']} - {note['pitch']} (면적: {note['area']:.1f})")
    else:
        print("\n❌ 음표가 검출되지 않았습니다.")
    
    # 결과 저장
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
    
    with open('improved_omr_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 결과 저장: improved_omr_result.json")

if __name__ == "__main__":
    test_improved_omr()
