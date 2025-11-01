# Fitzy 앱 실행 가이드 🚀

## ✅ 업데이트 완료 사항

### 학습된 모델 적용
- ✅ 30 epochs 학습된 패션 모델 (`yolov5_fashion.pt`) 적용 완료
- ✅ 13개 패션 클래스 탐지 지원
- ✅ 영어 클래스 이름 → 한국어 자동 변환

### 탐지 가능한 패션 아이템 (13종)
1. **긴팔 드레스** (long sleeve dress)
2. **긴팔 아우터** (long sleeve outwear)
3. **긴팔 상의** (long sleeve top)
4. **반팔 드레스** (short sleeve dress)
5. **반팔 아우터** (short sleeve outwear)
6. **반팔 상의** (short sleeve top)
7. **반바지** (shorts)
8. **스커트** (skirt)
9. **끈 드레스** (sling dress)
10. **끈 상의** (sling)
11. **바지** (trousers)
12. **조끼 드레스** (vest dress)
13. **조끼** (vest)

---

## 📋 사전 준비

### 1. 가상환경 확인

```bash
# 프로젝트 디렉토리로 이동
cd /Users/jimin/opensw/FItzy_copy

# 가상환경 활성화 (macOS/Linux)
source fitzy_env/bin/activate

# 또는 Windows
# fitzy_env\Scripts\activate
```

### 2. 필요한 패키지 설치 확인

```bash
# requirements.txt에 있는 패키지 설치
pip install -r requirements.txt
```

**주요 패키지:**
- `streamlit`: 웹 UI
- `torch`: PyTorch
- `ultralytics`: YOLOv5
- `transformers`: CLIP
- `rembg`: 배경 제거
- `mediapipe`: 얼굴/체형 분석

### 3. 모델 파일 확인

```bash
# 학습된 패션 모델 확인
ls -lh models/weights/yolov5_fashion.pt

# 예상 출력: -rw-r--r-- ... 9.9M ... yolov5_fashion.pt
```

---

## 🚀 앱 실행 방법

### 기본 실행

```bash
# 가상환경 활성화 후
streamlit run app.py
```

**실행 후:**
- 브라우저가 자동으로 열립니다
- 기본 URL: `http://localhost:8501`
- 터미널에 모델 로드 메시지가 표시됩니다

### 특정 포트 지정

```bash
streamlit run app.py --server.port 8502
```

### 헤드리스 모드 (브라우저 자동 열기 방지)

```bash
streamlit run app.py --server.headless true
```

---

## 🎯 사용 방법

### 1. 이미지 업로드
- **"이미지 업로드"** 섹션에서 사진 업로드
- 지원 형식: JPG, PNG, JPEG
- **배경 제거는 자동으로 수행됩니다**

### 2. 설정 선택
사이드바에서:
- **MBTI 유형**: ENFP, ISTJ, ESFP, INTJ, 기타
- **성별**: 남성, 여성, 공용
- **날씨 정보**: 온도, 날씨 상태, 계절

### 3. 분석 실행
- **"코디 분석하기"** 버튼 클릭
- YOLO가 13개 패션 아이템 자동 탐지
- CLIP이 스타일/색상 분석
- 얼굴/체형 분석 (자동)

### 4. 결과 확인
- **🎨 코디 추천**: 스타일별 추천 의상
- **🧭 현재 코디 평가**: 점수 및 피드백
- **👤 체형 맞춤 추천**: 개인화된 추천
- **🔍 진단 모드**: YOLO/CLIP 상세 분석 (토글 활성화)

---

## 🔧 문제 해결

### 모델 로드 실패

**증상:**
```
모델 파일이 없습니다: models/weights/yolov5_fashion.pt
사전 학습된 YOLOv5 모델을 사용합니다: yolov5n
```

**해결:**
```bash
# 모델 파일 확인
ls -lh models/weights/yolov5_fashion.pt

# 파일이 없으면 학습된 모델 복사
cp yolo5_fashion2/weights/best.pt models/weights/yolov5_fashion.pt
```

### CLIP 모델 로드 오류

**증상:**
```
NotImplementedError: Cannot copy out of meta tensor
```

**해결:**
- 이미 수정됨: `src/models/models.py`에서 CPU 먼저 로드 후 device 이동

### rembg 라이브러리 오류

**증상:**
```
⚠️ rembg 라이브러리가 없어 원본 이미지로 분석합니다.
```

**해결:**
```bash
pip install rembg
```

### 메모리 부족

**증상:**
- 앱 실행 중 메모리 오류
- 이미지 처리 중 중단

**해결:**
- 이미지 크기 줄이기
- 배치 처리 비활성화 (이미 적용됨)

---

## 📊 성능 확인

### 모델 로드 확인

앱 시작 시 다음 메시지가 표시되어야 합니다:

```
✅ YOLOv5 패션 모델 로드 완료: models/weights/yolov5_fashion.pt
✅ 탐지 가능한 클래스: ['long sleeve dress', 'long sleeve outwear', ...]
✅ CLIP 모델 로드 완료: openai/clip-vit-base-patch32
✅ 패션 추천 시스템 초기화 완료!
```

### 탐지 성능

- **mAP50**: 48.2% (실용 수준)
- **Precision**: 56.5%
- **Recall**: 48.7%

**참고**: 30 epochs까지 학습 완료. 추가 학습 시 성능 향상 가능.

---

## 🎨 주요 기능

### 1. 자동 배경 제거
- 업로드한 이미지의 배경 자동 제거
- 의류만 선명하게 분석

### 2. 패션 아이템 탐지
- 13개 패션 클래스 자동 탐지
- 바운딩 박스 표시 (진단 모드)

### 3. 스타일/색상 분석
- CLIP 기반 스타일 분석
- 색상 유사도 점수 제공

### 4. 얼굴/체형 분석
- MediaPipe 기반 얼굴 분석
- 체형 분류 (어깨/힙 비율)

### 5. 개인화 추천
- MBTI 기반 스타일 추천
- 성별 맞춤 제품 추천
- 체형 맞춤 코디 추천

### 6. 점수 시스템
- 외모 점수 (0-100)
- 패션 점수 (0-100)
- 상세 피드백 제공

---

## 🛑 앱 종료

터미널에서:
```bash
Ctrl + C
```

또는 브라우저에서:
- Streamlit 설정 메뉴 → **"Always rerun"** 해제
- 또는 탭 닫기

---

## 📝 다음 단계

### 추가 학습 (선택)
100 epochs까지 학습하려면:

```bash
python train_fashion.py \
  --resume \
  --resume-from yolo5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

**예상 개선**: mAP50 48% → 55-60%

---

## 🔗 관련 문서

- **학습 결과 검토**: `TRAINING_REVIEW_30EPOCHS.md`
- **학습 가이드**: `DATASET_USAGE_GUIDE.md`
- **모델 관리**: 앱 내 **"⚙️ 모델 관리"** 탭

---

**문의사항이 있으면 이슈를 등록하거나 코드를 검토해주세요!** 🎉

