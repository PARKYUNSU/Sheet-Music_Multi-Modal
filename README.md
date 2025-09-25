# 🎶 실시간 악보-연주 싱크 & 가사 디스플레이

오디오 신호 처리와 Optical Music Recognition(OMR)을 활용한 실시간 악보-연주 싱크 시스템입니다.

## 🚀 프로젝트 개요

이 프로젝트는 다음과 같은 기능을 제공합니다:

- **OMR (Optical Music Recognition)**: 스캔된 악보를 디지털 음악 데이터로 변환
- **실시간 오디오 처리**: 마이크 입력을 통한 실시간 음성 인식
- **악보-연주 싱크**: 연주와 악보의 실시간 동기화
- **가사 디스플레이**: 음악과 함께 가사 표시
- **웹 기반 UI**: 실시간 인터랙티브 인터페이스

## 🛠️ 환경 설정

### 1. Conda 환경 활성화

```bash
# 새로운 conda 환경 생성 (이미 완료됨)
conda create -n sheet-music python=3.10 -y

# 환경 활성화
conda activate sheet-music
```

### 2. 시스템 의존성 설치

```bash
# macOS에서 필요한 시스템 라이브러리
brew install portaudio tesseract

# Java (Audiveris용)
brew install openjdk@17
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
```

### 3. Python 라이브러리 설치

```bash
# 프로젝트 디렉토리로 이동
cd /Users/parkyunsu/project/Sheet-Music_Multi-Modal

# 모든 의존성 설치
pip install -r requirements.txt
```

## 📁 프로젝트 구조

```
Sheet-Music_Multi-Modal/
├── cv_model.py              # 기존 기본 OMR 코드
├── advanced_omr.py          # 고급 OMR 시스템
├── omr_integration.py       # 통합 OMR 모듈
├── requirements.txt         # Python 의존성
├── sheet music_1.png       # 테스트용 악보 이미지
├── preprocessed_image.png   # 전처리된 이미지
├── omr_result.json         # OMR 처리 결과
├── output.mid              # 생성된 MIDI 파일
└── audiveris/              # Audiveris OMR 엔진
```

## 🎯 사용법

### 1. 기본 OMR 테스트

```bash
# 환경 활성화
conda activate sheet-music

# OMR 시스템 실행
python omr_integration.py
```

### 2. 개별 모듈 테스트

```bash
# 고급 OMR만 테스트
python advanced_omr.py

# 기존 OMR 테스트
python cv_model.py
```

## 📊 현재 성능

- **처리 시간**: ~0.05초 (매우 빠름)
- **신뢰도**: 0.70 (중간 수준)
- **오선 검출**: 4개 오선 그룹 성공
- **음표 검출**: 개선 필요 (현재 0개)

## 🔧 설치된 주요 라이브러리

### 이미지 처리
- `opencv-python`: 컴퓨터 비전
- `Pillow`: 이미지 처리
- `pytesseract`: OCR (가사 인식)

### 음악 처리
- `music21`: 음악 분석 및 MusicXML 처리
- `pretty_midi`: MIDI 파일 생성
- `librosa`: 오디오 신호 처리

### 오디오 처리
- `pyaudio`: 실시간 오디오 입력
- `soundfile`: 오디오 파일 처리

### 웹 프레임워크
- `flask`: 웹 서버
- `flask-socketio`: 실시간 통신
- `websockets`: WebSocket 지원

### 데이터 처리
- `numpy`: 수치 계산
- `pandas`: 데이터 분석
- `matplotlib`: 시각화

## 🚧 개발 상태

### ✅ 완료된 기능
- [x] 새로운 conda 환경 설정
- [x] 모든 의존성 설치
- [x] 기본 OMR 시스템 구현
- [x] 고급 OMR 시스템 구현
- [x] 통합 OMR 모듈
- [x] MusicXML 분석
- [x] MIDI 생성

### 🔄 진행 중인 작업
- [ ] OMR 정확도 개선
- [ ] 음표 검출 알고리즘 강화
- [ ] 실시간 오디오 처리
- [ ] 악보-연주 싱크 알고리즘
- [ ] 웹 인터페이스 개발

### 📋 향후 계획
- [ ] 딥러닝 기반 음표 검출
- [ ] 실시간 피치 검출
- [ ] 템포 추정 및 조정
- [ ] 가사-음표 매핑
- [ ] 실시간 UI 개발

## 🐛 문제 해결

### PyAudio 설치 오류
```bash
# PortAudio 설치 후 재시도
brew install portaudio
pip install pyaudio
```

### Tesseract OCR 오류
```bash
# Tesseract 설치
brew install tesseract

# 한글 언어팩 설치 (선택사항)
brew install tesseract-lang
```

### Java 관련 오류
```bash
# Java 17 설치
brew install openjdk@17
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 연락처

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.

---

**Happy Coding! 🎵**