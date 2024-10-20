# 먼저 OpenCV를 설치하기 위해 다음 명령어를 터미널에 입력하세요:
# pip install opencv-python
# pip install pytesseract
# pip install Pillow

import cv2
import numpy as np
import pytesseract
import os
from PIL import Image
import re

# Tesseract 경로 설정 (기본 설치 경로가 아니라면 직접 설정)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# TESSDATA_PREFIX 설정
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'

def preprocess_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 바이너리 임계값을 사용하여 이미지 이진화
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    return binary

def detect_lines(binary_image):
    # 수평 및 수직 선을 검출하기 위해 구조 요소를 사용
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    
    # 수평선과 수직선을 각각 검출
    horizontal_lines = cv2.erode(binary_image, horizontal_structure)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_structure)
    
    vertical_lines = cv2.erode(binary_image, vertical_structure)
    vertical_lines = cv2.dilate(vertical_lines, vertical_structure)
    
    return horizontal_lines, vertical_lines

def detect_notes(binary_image):
    # 컨투어를 찾아 음표 검출
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notes = []
    for contour in contours:
        # 작은 노이즈를 무시하고 특정 크기 이상의 객체만 음표로 간주
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            notes.append((x, y, w, h))
            # 음표 영역을 사각형으로 그림
            cv2.rectangle(binary_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 화음 부분 처리 - 같은 X축 라인에 존재하는 음표 개수 확인
    note_positions = [note[0] for note in notes]
    unique_x_positions = set(note_positions)
    for x in unique_x_positions:
        count = note_positions.count(x)
        if count > 1:
            print(f"화음이 있는 X 좌표: {x}, 음표 개수: {count}")
    return notes

def remove_unwanted_symbols(img_original, template_paths):
    # 특정 기호들을 제거하기 위해 템플릿 매칭을 사용
    for template_path in template_paths:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        result = cv2.matchTemplate(img_original, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            bottom_right = (pt[0] + template.shape[1], pt[1] + template.shape[0])
            cv2.rectangle(img_original, pt, bottom_right, 255, -1)  # 흰색으로 덮어쓰기

def extract_staff_lines(binary_image):
    # 이미지 크기
    height, width = binary_image.shape
    
    # 각 줄의 오선 위치 검출
    line_positions = []
    limit = int(width * 0.8)
    
    for y in range(height):
        # 현재 줄의 검은색 픽셀(값이 0인 픽셀) 개수
        count = np.sum(binary_image[y] == 0)
        if count >= limit:
            line_positions.append(y)
    
    # 각 오선 첫 번째와 다섯 번째 줄 위치 검출
    one, five = [], []
    consecutive = 0
    
    for i in range(len(line_positions)):
        if i == 0 or line_positions[i] == line_positions[i - 1] + 1:
            consecutive += 1
        else:
            consecutive = 1
        
        if consecutive == 1:
            one.append(line_positions[i])
        elif consecutive == 5:
            five.append(line_positions[i])
    
    return one, five

def refine_notes_detection(binary_image):
    # 8분음표와 2분음표에 대한 인식 개선을 위해 침식 및 팽창 적용
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    binary_image = cv2.dilate(binary_image, kernel, iterations=3)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    return binary_image

def detect_time_signature(binary_image, notes):
    # 박자 인식 - 음표의 개수와 템플릿 매칭을 통해 박자 계산
    for note in notes:
        x, y, w, h = note
        if len(notes) > 1:
            # 음표가 여러 개 있는 경우 X축 좌표를 기준으로 박자 확인
            print(f"박자 인식: X좌표 {x}에서 박자 확인")
        else:
            # 기존 방식으로 박자 템플릿 매칭 수행
            print(f"기존 템플릿 매칭을 통해 박자 확인: X좌표 {x}")

def extract_lyrics(image_path, staff_lines):
    # 가사 추출 (OCR 사용)
    image = Image.open(image_path)
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, lang='kor', config=custom_oem_psm_config)
    
    # 불필요한 기호 및 특수문자 제거, 한글만 남기기
    lyrics = re.sub(r'[^가-힣\s]', '', text)
    lyrics = [line.strip() for line in lyrics.split('\n') if len(line.strip()) > 1]
    
    # 오선과 겹치지 않는 영역에서 가사만 추출
    lyrics_filtered = []
    for line in lyrics:
        if not any(staff_line for staff_line in staff_lines if isinstance(staff_line, int) and str(staff_line) in line):
            lyrics_filtered.append(line)
    
    return '\n'.join(lyrics_filtered)

def preprocess_for_ocr(image_path):
    # 이미지 전처리 (OCR 품질 개선)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 대비 조절 및 블러 적용
    image = cv2.convertScaleAbs(image, alpha=2.0, beta=50)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 바이너리화
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    # 전처리된 이미지 저장
    preprocessed_path = 'preprocessed_image.png'
    cv2.imwrite(preprocessed_path, binary)
    
    return preprocessed_path

def main(image_path, template_paths):
    # 이미지 전처리
    binary_image = preprocess_image(image_path)
    
    # 템플릿 매칭을 통해 불필요한 기호 제거 (샵, 플랫 등)
    remove_unwanted_symbols(binary_image, template_paths)
    
    # 오선 위치 검출
    one, five = extract_staff_lines(binary_image)
    
    # 소절 내 음표 검출 및 박자 인식
    for i in range(len(one)):
        rect = (0, max(0, one[i] - 20), binary_image.shape[1], min(binary_image.shape[0], five[i] - one[i] + 40))
        sub_image = binary_image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        
        # 소절 내 음표 검출
        sub_image = refine_notes_detection(sub_image)
        notes = detect_notes(sub_image)
        detect_time_signature(sub_image, notes)
    
    # 가사 추출 및 출력 (전처리 후 OCR 적용)
    preprocessed_image_path = preprocess_for_ocr(image_path)
    lyrics = extract_lyrics(preprocessed_image_path, one)
    print("추출된 가사:")
    print(lyrics)
    
    # 결과 이미지 보여주기
    cv2.imshow('Detected Notes', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지 경로와 템플릿 경로들을 지정하여 실행
main('sheet music_1.png', ['treble_clef_template.jpg', 'sharp_template.jpg', 'flat_template.jpg'])