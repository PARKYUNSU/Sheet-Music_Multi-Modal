"""
개선된 OMR 디버깅 스크립트
"""

import cv2
import numpy as np
from improved_omr import ImprovedOMR

def debug_improved_omr():
    """개선된 OMR 과정을 단계별로 디버깅"""
    
    print("🔍 개선된 OMR 디버깅 시작...")
    
    omr = ImprovedOMR()
    image_path = 'sheet music_1.png'
    
    # 1. 원본 이미지 로드
    print("\n1️⃣ 원본 이미지 로드")
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print(f"✅ 원본 이미지: {original_image.shape}")
    
    # 2. 기본 이진화
    print("\n2️⃣ 기본 이진화")
    binary_basic = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    print(f"✅ 기본 이진화 완료: {binary_basic.shape}")
    
    # 3. 오선 제거 과정
    print("\n3️⃣ 오선 제거 과정")
    
    # 수평선 검출을 위한 구조 요소
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    print(f"✅ 수평선 커널 생성: {horizontal_kernel.shape}")
    
    # 수평선 검출
    horizontal_lines = cv2.morphologyEx(binary_basic, cv2.MORPH_OPEN, horizontal_kernel)
    print(f"✅ 수평선 검출 완료")
    
    # 오선 제거
    binary_without_staff = binary_basic.copy()
    binary_without_staff[horizontal_lines > 0] = 0
    print(f"✅ 오선 제거 완료")
    
    # 4. 음표 강화
    print("\n4️⃣ 음표 강화")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary_without_staff, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    print(f"✅ 음표 강화 완료")
    
    # 5. 오선 검출 (원본 이진화 이미지로)
    print("\n5️⃣ 오선 검출")
    staff_groups = omr.detect_staff_lines_advanced(binary_basic)
    print(f"✅ 검출된 오선 그룹: {len(staff_groups)}개")
    
    for i, staff in enumerate(staff_groups):
        print(f"   오선 그룹 {i+1}: top={staff['top']}, bottom={staff['bottom']}")
    
    # 6. 각 단계별 컨투어 수 비교
    print("\n6️⃣ 각 단계별 컨투어 수 비교")
    
    # 원본 이진화
    contours_original, _ = cv2.findContours(binary_basic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   원본 이진화: {len(contours_original)}개 컨투어")
    
    # 오선 제거 후
    contours_no_staff, _ = cv2.findContours(binary_without_staff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   오선 제거 후: {len(contours_no_staff)}개 컨투어")
    
    # 음표 강화 후
    contours_enhanced, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   음표 강화 후: {len(contours_enhanced)}개 컨투어")
    
    # 7. 면적 분포 분석 (오선 제거 후)
    print("\n7️⃣ 오선 제거 후 면적 분포")
    areas = [cv2.contourArea(contour) for contour in contours_no_staff]
    areas = np.array(areas)
    
    if len(areas) > 0:
        print(f"   - 총 컨투어 수: {len(areas)}")
        print(f"   - 최소 면적: {areas.min():.1f}")
        print(f"   - 최대 면적: {areas.max():.1f}")
        print(f"   - 평균 면적: {areas.mean():.1f}")
        print(f"   - 중간값: {np.median(areas):.1f}")
        
        # 면적별 분포
        ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, float('inf'))]
        for min_area, max_area in ranges:
            count = np.sum((areas >= min_area) & (areas < max_area))
            percentage = (count / len(areas)) * 100 if len(areas) > 0 else 0
            print(f"   - {min_area}-{max_area}: {count}개 ({percentage:.1f}%)")
    else:
        print("   ❌ 컨투어가 없습니다!")
    
    # 8. 디버깅 이미지 저장
    print("\n8️⃣ 디버깅 이미지 저장")
    try:
        cv2.imwrite('debug_original_binary.png', binary_basic)
        cv2.imwrite('debug_horizontal_lines.png', horizontal_lines)
        cv2.imwrite('debug_no_staff.png', binary_without_staff)
        cv2.imwrite('debug_enhanced.png', enhanced)
        print("✅ 디버깅 이미지들 저장 완료")
    except Exception as e:
        print(f"❌ 이미지 저장 실패: {e}")
    
    # 9. 개선된 필터 조건으로 테스트
    print("\n9️⃣ 개선된 필터 조건 테스트")
    if len(staff_groups) > 0:
        staff = staff_groups[0]
        staff_region = enhanced[staff['top']-30:staff['bottom']+30, :]
        
        contours, _ = cv2.findContours(staff_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   첫 번째 오선 영역 컨투어: {len(contours)}개")
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 5000:  # 개선된 조건
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:
                    candidates.append({
                        'area': area,
                        'bbox': (x, y, w, h),
                        'aspect_ratio': aspect_ratio
                    })
        
        print(f"   개선된 필터 통과: {len(candidates)}개")
        
        for i, candidate in enumerate(candidates[:5]):  # 처음 5개만
            print(f"     {i+1}. 면적={candidate['area']:.1f}, "
                  f"bbox={candidate['bbox']}, "
                  f"비율={candidate['aspect_ratio']:.2f}")

if __name__ == "__main__":
    debug_improved_omr()
