"""
컨투어 면적 분석 스크립트 - 적절한 필터 조건 찾기
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from advanced_omr import AdvancedOMR

def analyze_contour_areas():
    """컨투어 면적 분포 분석"""
    
    print("📊 컨투어 면적 분석 시작...")
    
    omr = AdvancedOMR()
    image_path = 'sheet music_1.png'
    
    # 이미지 전처리
    binary_image = omr.preprocess_image_advanced(image_path)
    staff_groups = omr.detect_staff_lines_advanced(binary_image)
    
    all_areas = []
    all_contours_info = []
    
    print(f"총 {len(staff_groups)}개 오선 그룹 분석 중...")
    
    for i, staff in enumerate(staff_groups):
        print(f"\n오선 그룹 {i+1}:")
        
        # 오선 영역 추출
        staff_region = binary_image[staff['top']-20:staff['bottom']+20, :]
        
        # 컨투어 검출
        contours, _ = cv2.findContours(staff_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  - 검출된 컨투어 수: {len(contours)}")
        
        # 각 컨투어의 면적 분석
        for j, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            all_areas.append(area)
            all_contours_info.append({
                'staff': i+1,
                'contour_id': j+1,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': w/h if h > 0 else 0
            })
    
    # 면적 통계
    all_areas = np.array(all_areas)
    
    print(f"\n📈 전체 컨투어 면적 통계:")
    print(f"  - 총 컨투어 수: {len(all_areas)}")
    print(f"  - 최소 면적: {all_areas.min():.1f}")
    print(f"  - 최대 면적: {all_areas.max():.1f}")
    print(f"  - 평균 면적: {all_areas.mean():.1f}")
    print(f"  - 중간값: {np.median(all_areas):.1f}")
    print(f"  - 표준편차: {all_areas.std():.1f}")
    
    # 면적별 분포
    print(f"\n📊 면적별 분포:")
    ranges = [
        (0, 10, "매우 작음"),
        (10, 50, "작음"),
        (50, 100, "중간"),
        (100, 500, "큼"),
        (500, 1000, "매우 큼"),
        (1000, float('inf'), "거대")
    ]
    
    for min_area, max_area, label in ranges:
        count = np.sum((all_areas >= min_area) & (all_areas < max_area))
        percentage = (count / len(all_areas)) * 100
        print(f"  - {label} ({min_area}-{max_area}): {count}개 ({percentage:.1f}%)")
    
    # 현재 필터 조건으로 통과하는 컨투어
    current_filter_count = np.sum((all_areas > 50) & (all_areas < 1000))
    print(f"\n🔍 현재 필터 조건 (50 < area < 1000):")
    print(f"  - 통과하는 컨투어: {current_filter_count}개")
    
    # 개선된 필터 조건 제안
    print(f"\n💡 개선된 필터 조건 제안:")
    
    # 1. 더 관대한 조건
    relaxed_count = np.sum((all_areas > 10) & (all_areas < 2000))
    print(f"  - 관대한 조건 (10 < area < 2000): {relaxed_count}개")
    
    # 2. 중간값 기반 조건
    median_area = np.median(all_areas)
    median_based_count = np.sum((all_areas > median_area * 0.5) & (all_areas < median_area * 10))
    print(f"  - 중간값 기반 (median*0.5 < area < median*10): {median_based_count}개")
    
    # 3. 상위 20% 제외 조건
    percentile_80 = np.percentile(all_areas, 80)
    percentile_20 = np.percentile(all_areas, 20)
    percentile_based_count = np.sum((all_areas > percentile_20) & (all_areas < percentile_80))
    print(f"  - 백분위 기반 (20% < area < 80%): {percentile_based_count}개")
    
    # 상세 정보 출력 (면적이 큰 컨투어들)
    print(f"\n🔍 면적이 큰 컨투어들 (상위 10개):")
    sorted_contours = sorted(all_contours_info, key=lambda x: x['area'], reverse=True)
    
    for i, contour_info in enumerate(sorted_contours[:10]):
        print(f"  {i+1}. 오선{contour_info['staff']}-컨투어{contour_info['contour_id']}: "
              f"면적={contour_info['area']:.1f}, "
              f"bbox={contour_info['bbox']}, "
              f"비율={contour_info['aspect_ratio']:.2f}")
    
    return all_areas, all_contours_info

if __name__ == "__main__":
    analyze_contour_areas()
