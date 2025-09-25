"""
OMR 디버깅 스크립트 - 음표 검출 과정을 단계별로 분석
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from advanced_omr import AdvancedOMR

def debug_note_detection():
    """음표 검출 과정을 단계별로 디버깅"""
    
    print("🔍 OMR 디버깅 시작...")
    
    # 1. 이미지 로드 및 전처리
    print("\n1️⃣ 이미지 전처리 단계")
    omr = AdvancedOMR()
    image_path = 'sheet music_1.png'
    
    # 원본 이미지 로드
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print("❌ 이미지를 로드할 수 없습니다!")
        return
    
    print(f"✅ 원본 이미지 로드: {original_image.shape}")
    
    # 전처리된 이미지
    binary_image = omr.preprocess_image_advanced(image_path)
    print(f"✅ 이진화 완료: {binary_image.shape}")
    
    # 2. 오선 검출
    print("\n2️⃣ 오선 검출 단계")
    staff_groups = omr.detect_staff_lines_advanced(binary_image)
    print(f"✅ 검출된 오선 그룹 수: {len(staff_groups)}")
    
    for i, staff in enumerate(staff_groups):
        print(f"   오선 그룹 {i+1}: top={staff['top']}, bottom={staff['bottom']}")
    
    # 3. 각 오선에서 음표 검출 시도
    print("\n3️⃣ 음표 검출 단계")
    total_candidates = 0
    total_notes = 0
    
    for i, staff in enumerate(staff_groups):
        print(f"\n   오선 그룹 {i+1} 분석:")
        
        # 오선 영역 추출
        staff_region = binary_image[staff['top']-20:staff['bottom']+20, :]
        print(f"   - 오선 영역 크기: {staff_region.shape}")
        
        # 컨투어 검출
        contours, _ = cv2.findContours(staff_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   - 검출된 컨투어 수: {len(contours)}")
        
        # 음표 후보 필터링
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000:  # 현재 필터 조건
                x, y, w, h = cv2.boundingRect(contour)
                candidates.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        print(f"   - 필터링된 후보 수: {len(candidates)}")
        total_candidates += len(candidates)
        
        # 각 후보의 상세 정보
        for j, candidate in enumerate(candidates):
            x, y, w, h = candidate['bbox']
            area = candidate['area']
            print(f"     후보 {j+1}: bbox=({x},{y},{w},{h}), area={area:.1f}")
        
        # 음표 분석
        notes_in_staff = 0
        for candidate in candidates:
            note_info = omr.analyze_note_candidate(candidate, staff)
            if note_info:
                notes_in_staff += 1
                print(f"     ✅ 음표 검출: {note_info['type']}, pitch={note_info['pitch']}")
        
        print(f"   - 최종 음표 수: {notes_in_staff}")
        total_notes += notes_in_staff
    
    print(f"\n📊 전체 결과:")
    print(f"   - 총 컨투어 후보: {total_candidates}")
    print(f"   - 최종 음표 수: {total_notes}")
    
    # 4. 문제점 분석
    print("\n4️⃣ 문제점 분석")
    if total_candidates == 0:
        print("❌ 문제: 컨투어 후보가 전혀 검출되지 않음")
        print("   가능한 원인:")
        print("   - 이진화 임계값이 너무 높음/낮음")
        print("   - 모폴로지 연산이 음표를 제거함")
        print("   - 오선이 음표를 가리고 있음")
    elif total_notes == 0:
        print("❌ 문제: 컨투어는 검출되지만 음표로 인식되지 않음")
        print("   가능한 원인:")
        print("   - 음표 분류 알고리즘이 너무 엄격함")
        print("   - 음표 크기 필터가 부적절함")
        print("   - 음표 후보 분석 로직에 문제")
    else:
        print("✅ 음표 검출이 정상적으로 작동함")
    
    # 5. 시각화 (선택사항)
    print("\n5️⃣ 디버깅 이미지 저장")
    try:
        # 원본 이미지에 오선 표시
        debug_image = original_image.copy()
        for i, staff in enumerate(staff_groups):
            cv2.rectangle(debug_image, (0, staff['top']-20), (debug_image.shape[1], staff['bottom']+20), (0, 255, 0), 2)
            cv2.putText(debug_image, f"Staff {i+1}", (10, staff['top']-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite('debug_staff_detection.png', debug_image)
        print("✅ debug_staff_detection.png 저장됨")
        
        # 이진화 이미지 저장
        cv2.imwrite('debug_binary_image.png', binary_image)
        print("✅ debug_binary_image.png 저장됨")
        
    except Exception as e:
        print(f"❌ 이미지 저장 실패: {e}")

if __name__ == "__main__":
    debug_note_detection()
