# 🛠️ 환경 설정 가이드

## 새로운 Conda 환경 설정 완료!

### ✅ 완료된 작업들

1. **기존 환경 정리**: 의존성 충돌 해결
2. **새 conda 환경 생성**: `sheet-music` 환경 생성
3. **시스템 의존성 설치**: PortAudio, Tesseract, Java 17
4. **Python 라이브러리 설치**: 모든 필요한 패키지 설치
5. **테스트 완료**: OMR 시스템 정상 작동 확인

### 🎯 환경 활성화 방법

```bash
# 환경 활성화
conda activate sheet-music

# 현재 환경 확인
conda info --envs
```

### 📦 설치된 주요 패키지들

#### 이미지 처리 & 컴퓨터 비전
- `opencv-python==4.12.0.88`
- `Pillow==11.3.0`
- `pytesseract==0.3.13`

#### 음악 처리
- `music21==9.7.1`
- `pretty_midi==0.2.10`
- `librosa==0.11.0`

#### 오디오 처리
- `pyaudio==0.2.14`
- `soundfile==0.13.1`

#### 웹 프레임워크
- `flask==3.1.2`
- `flask-socketio==5.5.1`
- `websockets==15.0.1`

#### 데이터 처리
- `numpy==2.2.6`
- `pandas==2.3.2`
- `matplotlib==3.10.6`
- `scipy==1.15.3`

### 🚀 빠른 시작

```bash
# 1. 환경 활성화
conda activate sheet-music

# 2. 프로젝트 디렉토리로 이동
cd /Users/parkyunsu/project/Sheet-Music_Multi-Modal

# 3. OMR 시스템 실행
python omr_integration.py
```

### 🔧 환경 관리 명령어

```bash
# 환경 비활성화
conda deactivate

# 환경 삭제 (필요시)
conda env remove -n sheet-music

# 환경 재생성 (필요시)
conda create -n sheet-music python=3.10 -y
conda activate sheet-music
pip install -r requirements.txt
```

### 📊 현재 시스템 상태

- **Python 버전**: 3.10.18
- **Conda 환경**: sheet-music
- **의존성 충돌**: 해결됨
- **OMR 시스템**: 정상 작동
- **처리 속도**: ~0.05초 (매우 빠름)

### 🎵 다음 단계

이제 깔끔한 환경에서 다음 작업들을 진행할 수 있습니다:

1. **OMR 정확도 개선**
2. **실시간 오디오 처리 모듈 개발**
3. **악보-연주 싱크 알고리즘 구현**
4. **웹 인터페이스 개발**

환경 설정이 완료되었습니다! 🎉
