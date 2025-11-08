# Fitzy - AI 패션 코디 추천 시스템

## 📋 프로젝트 개요

Fitzy는 AI 기술을 활용한 개인화 패션 코디 추천 시스템입니다. 사용자가 업로드한 옷 사진을 분석하고, MBTI, 계절, 날씨 정보를 바탕으로 최적의 코디를 추천하며, 가상 피팅을 통해 업로드한 이미지에 추천 코디를 실제로 합성하여 시각적으로 보여줍니다.

**개발 환경**: macOS (Apple Silicon M2), Python 3.12

---

## 🎯 주요 기능

### 1. 이미지 기반 의류 분석
- **YOLOv5 기반 의류 탐지**: 업로드된 이미지에서 의류 아이템 자동 감지
- **CLIP 기반 스타일 분석**: 의류의 색상, 스타일, 패턴 분석
- **CLIP 후처리 검증**: YOLO 탐지 결과를 CLIP으로 재검증하여 정확도 향상

### 2. 개인화 추천
- **MBTI 기반 스타일 매칭**: 사용자의 MBTI 유형에 맞는 스타일 추천
- **날씨/계절 고려**: 현재 날씨와 계절에 적합한 코디 제안
- **얼굴/체형 분석**: MediaPipe 기반 얼굴 형태 및 체형 분석으로 개인 맞춤 추천

### 3. 다차원 점수 평가
- **외모 점수**: 얼굴, 체형 종합 평가
- **패션 점수**: 아이템 구성, 스타일 일치도, 계절/날씨 적합성 평가
- **향상된 평가 시스템**: 색상 분포, K-means 클러스터링 기반 고급 분석

### 4. 가상 피팅
- **Stable Diffusion Inpainting**: 업로드한 이미지에 추천 코디를 실제로 합성
- **MPS 최적화**: Apple Silicon (M1/M2) GPU 가속 지원
- **DPM Solver 스케줄러**: 빠르고 안정적인 이미지 생성 (12 steps)
- **자연스러운 합성**: 의류 질감, 조명, 그림자까지 자연스럽게 생성

### 5. 텍스트 기반 검색
- **키워드 검색**: "파티용 코디", "출근룩" 등 텍스트 기반 검색
- **CLIP 기반 의미론적 검색**: 이미지-텍스트 매칭

### 6. 모델 관리
- **모델 상태 모니터링**: YOLO, CLIP 모델 상태 확인
- **시스템 정보**: CPU, 메모리, 디스크 사용량 확인
- **학습 관리**: 학습 상태 확인 (향후 확장 예정)

---

## 🛠️ 기술 스택

### 프레임워크
- **Streamlit 1.50.0**: 웹 UI 프레임워크
- **PyTorch 2.8.0**: 딥러닝 프레임워크
- **Transformers 4.57.0**: Hugging Face 트랜스포머 라이브러리

### AI 모델
- **YOLOv5**: 의류 객체 탐지
  - 기본 모델: `yolov5n.pt`
  - 커스텀 모델: `yolov5_fashion.pt` (DeepFashion2 데이터셋으로 학습, 30 epochs)
  - 탐지 클래스: 상의, 하의, 드레스, 재킷 등 패션 아이템
- **CLIP (openai/clip-vit-base-patch32)**: 이미지-텍스트 임베딩 및 스타일 분석
- **Stable Diffusion v1.4**: AI 코디 이미지 생성 (MPS 최적화)

### 컴퓨터 비전
- **MediaPipe 0.10.21**: 얼굴 및 체형 랜드마크 탐지
- **OpenCV 4.12**: 이미지 전처리 및 시각화
- **Pillow 11.3.0**: 이미지 처리

### 데이터 분석
- **Pandas 2.3.3**: 데이터 처리
- **NumPy 1.26.4**: 수치 연산
- **Scikit-learn**: K-means 클러스터링 (색상 분석)

### 기타 라이브러리
- **diffusers 0.35.2**: Stable Diffusion 파이프라인
- **accelerate 1.11.0**: 분산 학습 및 추론 가속
- **rembg 2.0.67**: 배경 제거 (선택적)
- **ultralytics 8.3.213**: YOLOv5/YOLOv8 통합 인터페이스

---

## 🏗️ 프로젝트 구조

```
FItzy/
├── app.py                          # 메인 Streamlit 애플리케이션
├── .fitzy_settings.json            # 설정값 영구 저장 파일
├── config.py                       # 설정 파일 (MBTI, 계절, 날씨 가이드)
├── requirements.txt                # Python 패키지 의존성
├── README.md                       # 프로젝트 README
│
├── src/
│   ├── models/
│   │   └── models.py              # AI 모델 클래스 (YOLO, CLIP, 추천 시스템)
│   │
│   └── utils/
│       ├── recommendation_engine.py   # 추천 엔진 (MBTI, 날씨, 계절 기반)
│       ├── virtual_fitting.py         # 가상 피팅 시스템 (Stable Diffusion Inpainting)
│       ├── scoring_system.py          # 외모/패션 점수 평가 시스템
│       ├── body_analysis.py           # 얼굴/체형 분석 (MediaPipe)
│       ├── visualization.py           # YOLO 탐지 결과 시각화
│       ├── model_manager.py           # 모델 상태 관리
│       └── common_utils.py            # 공통 유틸리티 (색상 변환, 디바이스 설정)
│
├── models/
│   └── weights/
│       ├── yolov5_fashion.pt      # 학습된 패션 모델 (30 epochs)
│       └── yolov5nu.pt             # 기본 YOLOv5 모델
│
├── data/                           # 학습 데이터셋 (DeepFashion2)
│
└── docs/                           # 프로젝트 문서
    ├── PROJECT_OVERVIEW.md         # 프로젝트 전체 개요 (본 문서)
    ├── VIRTUAL_FITTING_GUIDE.md    # 가상 피팅 사용 가이드
    ├── DATASET_USAGE_GUIDE.md      # 데이터셋 사용 가이드
    └── CURRENT_TRAINING_STATUS.md  # 학습 현황
```

---

## 💡 핵심 로직 및 알고리즘

### 1. 의류 탐지 파이프라인 (YOLODetector)

```python
# src/models/models.py - YOLODetector 클래스

1. 이미지 입력
   ↓
2. YOLOv5 모델로 객체 탐지 (Bounding Box + Class + Confidence)
   ↓
3. CLIP 기반 후처리 검증
   - 반팔/긴팔 등 세부 특징을 CLIP으로 재확인
   - 잘못된 분류 수정
   ↓
4. 탐지 결과 반환 (클래스명, 좌표, 신뢰도)
```

**주요 메서드**:
- `detect_clothes(image)`: 의류 탐지 및 CLIP 검증
- `_verify_with_clip()`: CLIP 기반 재검증

### 2. 스타일 분석 파이프라인 (CLIPAnalyzer)

```python
# src/models/models.py - CLIPAnalyzer 클래스

1. 이미지 입력
   ↓
2. CLIP 이미지 임베딩 생성
   ↓
3. 다양한 텍스트 프롬프트와 유사도 계산
   - 색상: "빨간색 옷", "파란색 옷", "검은색 옷" 등
   - 스타일: "캐주얼", "포멀", "트렌디" 등
   - 패턴: "체크 무늬", "스트라이프", "단색" 등
   ↓
4. 유사도 점수 기반 스타일 분석 결과 반환
```

**주요 메서드**:
- `analyze_style(image)`: 전체 스타일 분석
- `calculate_similarity(image, texts)`: 이미지-텍스트 유사도 계산
- `detect_gender(image)`: CLIP 기반 성별 인식

### 3. 추천 엔진 (RecommendationEngine)

```python
# src/utils/recommendation_engine.py

추천 프로세스:
1. 사용자 입력 수집
   - MBTI 유형
   - 날씨 정보 (온도, 날씨 상태)
   - 계절 정보
   - 이미지 분석 결과 (선택적)
   ↓
2. MBTI 스타일 매칭
   - MBTI별 선호 스타일, 색상, 패턴 매핑
   ↓
3. 계절/날씨 기반 필터링
   - 온도에 따른 소재 선택
   - 날씨에 따른 액세서리 추천
   ↓
4. 이미지 분석 결과 반영
   - 탐지된 아이템과 조화로운 아이템 추천
   - CLIP 분석 색상/스타일과 매칭
   ↓
5. 3가지 버전 코디 생성 (각각 다른 색상/스타일)
   - 버전 1: MBTI 기반 (MBTI 색상 + 대비 색상)
   - 버전 2: 계절 기반 (계절 색상 팔레트)
   - 버전 3: 이미지/날씨 기반 (계절 색상의 다른 조합)
   ↓
6. 구체적 제품 추천
   - 브랜드별 추천 제품 매칭
   ↓
7. 추천 이유 생성 (MBTI/계절/날씨 연계)
   - MBTI 유형별 특성 설명
   - 계절별 색상/소재 설명
   - 날씨별 적합성 설명
   - 온도별 레이어링 설명
   - 이미지 분석 결과 반영
```

**주요 메서드**:
- `generate_unified_outfit_recommendations()`: 통합 추천 생성 (성별 + MBTI + 이미지 분석 + 온도/계절)
- `get_personalized_recommendation()`: 개인화 추천 (기존 호환성 유지)
- `_generate_recommendation_reason()`: MBTI/계절/날씨 연계 추천 이유 생성
- `_generate_product_recommendations()`: 아이템 기반 제품 추천 (아이템에 정확히 맞는 제품)
- `evaluate_current_outfit()`: 현재 코디 평가 (추천 이유와 연계)
- `_get_color_from_palette()`: MBTI와 계절을 고려한 색상 선택
- `_translate_color_to_korean()`: 색상 텍스트를 한글 색상명으로 변환

### 4. 얼굴/체형 분석 (BodyAnalyzer)

```python
# src/utils/body_analysis.py

얼굴 분석:
1. MediaPipe FaceMesh로 468개 랜드마크 추출
   ↓
2. 얼굴 비율 계산 (가로/세로)
   ↓
3. 얼굴 형태 분류
   - 둥근형: 비율 < 1.15
   - 계란형: 1.15 ≤ 비율 < 1.30
   - 길쭉한형: 비율 ≥ 1.30
   ↓
4. 눈 크기 분석 (랜드마크 기반)

체형 분석:
1. MediaPipe Pose로 33개 신체 랜드마크 추출
   ↓
2. 어깨/허리 비율 계산
   ↓
3. 체형 분류
   - 균형잡힌 체형: 비율 < 1.3
   - 어깨가 넓은 체형: 비율 ≥ 1.3
```

**주요 메서드**:
- `analyze_face(image)`: 얼굴 분석
- `analyze_body(image)`: 체형 분석
- `detect_gender(image)`: 성별 인식 (DeepFace 선택적)

### 5. 점수 평가 시스템 (ScoringSystem)

```python
# src/utils/scoring_system.py

외모 점수:
- 얼굴 점수: 얼굴 탐지 여부, 비율, 대칭성
- 체형 점수: 체형 탐지 여부, 비율
- 전체 외모: 얼굴 + 체형의 가중 평균

패션 점수:
- 아이템 구성: 탐지된 아이템 수, 다양성
- 스타일 일치도: CLIP 유사도 기반
- 계절 적합성: 계절별 색상/소재 매칭
- 날씨 적합성: 온도/날씨에 따른 의류 적합성
- 전체 패션: 종합 점수

향상된 평가 (EnhancedScoringSystem):
- K-means 클러스터링 기반 색상 분포 분석
- CLIP 임베딩 기반 고급 스타일 분석
```

### 6. 가상 피팅 (VirtualFittingSystem)

```python
# src/utils/virtual_fitting.py

가상 피팅 프로세스:
1. YOLO로 의류 영역 탐지 (상의/하의)
   ↓
2. 탐지된 영역을 마스크로 변환
   ↓
3. Stable Diffusion 2 Inpainting 파이프라인 로드
   - UNet: MPS (GPU)
   - VAE: CPU
   - TextEncoder: CPU
   - Scheduler: DPM Solver Multistep (빠르고 안정적)
   ↓
4. 프롬프트 생성 (색상/타입 정확히 명시, 공통 유틸리티 사용)
   - "a man wearing a navy short sleeve t-shirt, EXACTLY navy color"
   - "realistic fit, naturally worn, proper draping"
   - 공통 유틸리티(`extract_color_from_text`)로 색상 추출
   ↓
5. Inpainting 실행 (11 inference steps for MPS, 7 for CPU, guidance_scale 7.5)
   - 마스크 영역만 새로운 의류로 교체
   - 나머지 영역은 원본 유지
   ↓
6. 자연스러운 합성 결과 반환 (원본 크기 유지)
   - 반환값: (이미지, 프롬프트 정보) 튜플
   - 프롬프트 정보: 각 영역(상의/하의)별 프롬프트 포함
```

**최적화 포인트**:
- **MPS 백엔드**: Apple Silicon GPU 가속 (공통 유틸리티 사용)
- **DPM Solver 스케줄러**: PNDM 대신 사용 (더 빠르고 안정적)
- **스텝 수 최적화**: 20 → 11 steps (MPS) 또는 7 steps (CPU) (속도 향상, IndexError 방지)
- **지연 로딩**: 첫 사용 시에만 모델 로드
- **합성 시간**: 약 9-10초 (아이템당, MPS 기준)
- **프롬프트 정보 반환**: 각 영역별 프롬프트 정보 포함하여 사용자에게 표시

---

## 🔧 주요 클래스 및 파일

### `app.py`
메인 Streamlit 애플리케이션

**주요 함수**:
- `main()`: 메인 애플리케이션 로직
- `display_outfit_recommendations()`: 코디 추천 결과 표시
  - 추천 코디 1: 자동 생성
  - 추천 코디 2, 3: 버튼 클릭으로 생성
  - 프롬프트 정보 표시 (fold 상태)
- `display_text_search_results()`: 텍스트 검색 결과 표시
  - 성별별 아이템 반환
  - 가상 피팅 이미지 생성 (버튼 클릭)
  - 프롬프트 정보 표시 (fold 상태)
- `display_trend_outfits()`: 트렌드 코디 표시
- `display_model_manager()`: 모델 관리자 페이지
- `render_gender_selector()`: 성별 선택 UI (자동 인식 포함)
- `render_outfit_items_display()`: 코디 아이템 표시
- `display_score_metric()`: 점수 메트릭 표시
- `detect_gender_from_image()`: 이미지 기반 성별 인식
- `load_settings()`: 설정값 파일에서 로드
- `save_settings()`: 설정값 파일에 저장

### `src/models/models.py`
AI 모델 통합 클래스

**클래스**:
- `YOLODetector`: YOLOv5 기반 의류 탐지
- `CLIPAnalyzer`: CLIP 기반 스타일 분석
- `WeatherBasedRecommender`: 날씨 기반 추천
- `MBTIAnalyzer`: MBTI 기반 스타일 분석
- `TextBasedSearcher`: CLIP 기반 텍스트 검색
  - 성별별 아이템 반환 (남성/여성 구분)
  - 카테고리별 아이템 매핑 (파티용, 출근룩, 데이트룩)
- `FashionRecommender`: 통합 패션 추천 시스템

### `src/utils/recommendation_engine.py`
추천 로직 엔진

**주요 기능**:
- 개인화 추천 생성
- 제품 추천
- 현재 코디 평가
- MBTI/계절/날씨 연계 추천 이유 생성

### `src/utils/virtual_fitting.py`
가상 피팅 시스템

**클래스**:
- `VirtualFittingSystem`: Stable Diffusion Inpainting 기반 가상 피팅
  - YOLO 기반 의류 영역 탐지
  - Inpainting으로 실제 의류 합성
  - MPS 최적화 (공통 유틸리티 사용)
  - DPM Solver 스케줄러 사용 (11 steps for MPS, 7 steps for CPU)
  - 프롬프트 정보 반환 (이미지, 프롬프트 정보 튜플)
  - 색상 변환 로직 (공통 유틸리티 사용)

### `src/utils/scoring_system.py`
점수 평가 시스템

**클래스**:
- `ScoringSystem`: 기본 점수 평가
  - `score_appearance()`: 외모 점수 평가
  - `score_fashion()`: 패션 점수 평가
  - `get_score_label()`: 점수 레이블 반환
  - `get_detailed_feedback()`: 상세 피드백 생성

### `src/utils/common_utils.py`
공통 유틸리티 함수

**주요 함수**:
- `get_device_info()`: 디바이스 정보 반환 (MPS 우선, CPU 폴백)
- `extract_color_from_text()`: 텍스트에서 색상 추출 (한글/영어)
- `extract_color_bgr()`: 텍스트에서 색상 추출 (BGR 형식)
- `translate_color_to_english()`: 한글 색상명을 영어로 변환
- `COLOR_MAP`: 색상 변환 맵 (한글 → 영어)

### `src/utils/body_analysis.py`
얼굴/체형 분석

**클래스**:
- `BodyAnalyzer`: MediaPipe 기반 얼굴/체형 분석

### `config.py`
설정 및 매핑 데이터

**주요 데이터**:
- `MBTI_STYLES`: MBTI별 스타일, 색상, 패턴 매핑
- 계절별 가이드
- 날씨별 가이드

---

## 🔄 데이터 흐름

```
사용자 입력 (이미지 + MBTI + 날씨 + 계절)
         ↓
┌────────────────────────────────────────┐
│  이미지 분석 (YOLODetector + CLIPAnalyzer)  │
│  - 의류 탐지                              │
│  - 스타일/색상 분석                        │
│  - 성별 인식                              │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  얼굴/체형 분석 (BodyAnalyzer)             │
│  - 얼굴 형태 분석                         │
│  - 체형 분석                              │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  점수 평가 (ScoringSystem)                │
│  - 외모 점수                              │
│  - 패션 점수                              │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  추천 생성 (RecommendationEngine)         │
│  - MBTI 기반 추천                         │
│  - 계절 기반 추천                         │
│  - 날씨 기반 추천                         │
│  - 이미지 분석 결과 반영                   │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  가상 피팅 (VirtualFittingSystem)          │
│  - Stable Diffusion Inpainting 실행      │
│  - 업로드 이미지에 코디 합성              │
│  - MPS GPU 가속                          │
└────────────────────────────────────────┘
         ↓
   추천 코디 결과 (3가지 버전)
   + 가상 피팅 이미지
   + 구체적 제품 추천
   + MBTI/계절 연계 추천 이유
```

---

## 🚀 주요 기술적 성과

### 1. MPS (Metal Performance Shaders) 최적화
- Apple Silicon M1/M2에서 GPU 가속 지원
- 디바이스별 최적 배치:
  - UNet: MPS (연산 집약적)
  - VAE: CPU (타입 호환성)
  - TextEncoder: CPU (임베딩 안정성)
- 생성 시간: CPU 대비 약 5배 향상

### 2. CLIP 기반 다차원 분석
- 색상 분석: 20+ 색상 키워드 유사도 계산
- 스타일 분석: 캐주얼, 포멀, 트렌디 등
- 패턴 분석: 체크, 스트라이프, 단색 등
- 의미론적 검색: 텍스트-이미지 매칭

### 3. 커스텀 YOLOv5 모델 학습
- 데이터셋: DeepFashion2 Small-32k (32,081 images)
- 학습 결과: 30 epochs 완료
- 클래스: 상의, 하의, 드레스, 재킷 등 13개 카테고리
- 정확도: mAP 개선 (학습 진행 중)

### 4. 코드 간소화 및 리팩토링
- **사용하지 않는 코드 제거**:
  - `image_generator.py` 삭제 (사용되지 않음)
  - `enhanced_scorer` 관련 코드 제거 (항상 비활성화)
- **공통 유틸리티 생성**:
  - `common_utils.py` 추가: 색상 변환, 디바이스 설정 등 재사용 가능한 함수
  - 중복된 색상 변환 로직 통합
  - 디바이스 설정 로직 통합
- **기능 개선**:
  - 설정값 영구 저장 기능 추가 (`.fitzy_settings.json`)
  - 추천 이유 생성 로직 개선 (MBTI/계절/날씨 연계 상세 설명)
  - "이 조합이 어울리는 이유"와 "현재 코디 평가" 통합
  - 프롬프트 정보 표시 기능 추가 (fold 상태)
- **UI/UX 개선**:
  - 추천 코디 1은 자동 생성, 2와 3은 버튼 클릭으로 생성
  - 텍스트 검색에서 성별별 아이템 반환
  - 이미지 생성 중 다른 탭 블로킹 방지

---

## 📊 성능 지표

### 모델 성능
- **YOLOv5 추론 시간**: 약 0.1-0.3초
- **CLIP 분석 시간**: 약 0.2-0.5초
- **전체 분석 시간**: 약 1-2초

### 가상 피팅
- **첫 로드 시간**: 약 5-10초 (모델 로드)
- **합성 시간**: 약 9-10초 (아이템당, MPS 기준, 11 inference steps)
- **이미지 크기**: 원본 크기 유지 (최대 512px로 리사이즈)
- **메모리 사용**: 약 5GB (모델 크기)
- **프롬프트 정보**: 각 이미지 생성 시 사용된 프롬프트 표시 (fold 상태)

### 시스템 요구사항
- **최소 메모리**: 8GB RAM
- **권장 메모리**: 16GB RAM
- **GPU**: Apple Silicon (M1/M2) 필수 (MPS 지원)
- **디스크 공간**: 약 5GB (모델 + 데이터셋)

---

## 🎨 UI/UX 특징

### 메인 화면 구성
1. **사이드바**:
   - MBTI 선택 (설정값 영구 저장)
   - 성별 선택 (자동 인식)
   - 진단 모드 (YOLO/CLIP 상세 분석)
   - 날씨 정보 입력 (온도, 날씨, 계절 - 설정값 영구 저장)

2. **이미지 분석 탭**:
   - 이미지 업로드
   - 얼굴/체형 분석 결과
   - 외모/패션 점수 (접힌 상태)
   - 추천 코디 3가지 (각각 다른 색상/스타일)
   - 가상 피팅 이미지:
     - 추천 코디 1: 자동 생성
     - 추천 코디 2, 3: 버튼 클릭으로 생성
     - 각 이미지에 사용된 프롬프트 정보 표시 (fold 상태)
   - 이 조합이 어울리는 이유 (MBTI/계절/날씨 연계 설명)
   - 현재 코디 평가 (추천 이유와 연계)

3. **텍스트 검색 탭**:
   - 키워드 검색 ("파티용 코디", "출근룩", "데이트룩" 등)
   - 빠른 선택 버튼
   - 성별별 아이템 반환 (남성/여성 구분)
   - 가상 피팅 이미지 생성 (버튼 클릭, 이미지 분석 섹션의 이미지와 설정값 사용)
   - 사용된 프롬프트 정보 표시 (fold 상태)

4. **트렌드 코디 탭**:
   - 계절별 인기 코디

5. **모델 관리 탭**:
   - 모델 상태 모니터링
   - 시스템 정보
   - 학습 관리 (준비 중)

### UX 개선사항
- 자동 성별 인식 및 실시간 반영
- 설정값 영구 저장 (MBTI, 온도, 날씨, 계절 - 서버 재시작 후에도 유지)
- 가상 피팅 캐시 기반 이미지 재사용 (동일 코디는 재생성 안 함)
- 진단 모드로 YOLO/CLIP 분석 결과 시각화
- MBTI/계절/날씨 연계 추천 이유 상세 설명
- 추천 코디 3개 각각 다른 색상/스타일로 다양성 확보
- 추천 코디 1은 자동 생성, 2와 3은 버튼 클릭으로 생성 (성능 최적화)
- 이미지 생성 중 다른 탭 블로킹 방지 (비동기 처리)
- 프롬프트 정보 표시 (사용자가 생성 과정 이해 가능)
- 텍스트 검색에서 성별별 맞춤 아이템 반환

---

## 🧪 진단 모드 기능

진단 모드를 활성화하면 다음 정보를 확인할 수 있습니다:

### YOLO 탐지 결과
- 바운딩 박스 시각화
- 탐지된 클래스명 (한글/영어)
- 신뢰도 점수
- 바운딩 박스 좌표 및 크기
- CLIP 검증으로 정정된 경우 표시

### CLIP 유사도 분석
- 색상 유사도 상위 10개
- 스타일 유사도 상위 10개
- 전체 유사도 차트 (Altair)
- 이미지 분석 점수

### 원시 결과 미리보기
- JSON 형식으로 원시 데이터 확인

---

## 📝 설정 파일 구조

### `config.py`

```python
MBTI_STYLES = {
    "ENFP": {
        "style": "활동적이고 자유로운",
        "colors": ["밝은 색상", "비비드한 색"],
        "patterns": ["프린트", "패턴 믹스"],
        "items": ["편한 티셔츠", "캐주얼 팬츠"]
    },
    # ... (4가지 MBTI 유형)
}

# 계절별 추천
# 날씨별 추천
# 온도별 가이드
```

---

## 🔬 학습된 모델

### YOLOv5 패션 모델
- **모델 파일**: `models/weights/yolov5_fashion.pt`
- **기반 모델**: YOLOv5n
- **학습 데이터**: DeepFashion2 Small-32k (32,081 images)
- **학습 Epochs**: 30 (부분 학습)
- **탐지 클래스**: 13개 패션 카테고리
  - 긴팔/반팔 상의, 드레스, 아우터
  - 긴/반바지, 치마
  - 신발, 액세서리 등

### CLIP 모델
- **모델**: `openai/clip-vit-base-patch32`
- **용도**: 이미지-텍스트 임베딩, 스타일 분석
- **장치**: CPU (안정성)

---

## 💻 실행 방법

### 1. 가상환경 설정
```bash
# 가상환경 생성
python -m venv fitzy_env

# 가상환경 활성화 (macOS)
source fitzy_env/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 애플리케이션 실행
```bash
streamlit run app.py
```

### 3. 웹 브라우저 접속
```
Local URL: http://localhost:8501
```

---

## 🔒 제약사항 및 알려진 이슈

### 제약사항
1. **MPS 필수**: Apple Silicon (M1/M2) 맥북에서만 가상 피팅 가능
2. **첫 실행 시간**: Stable Diffusion Inpainting 모델 다운로드 (약 5GB, 수 분 소요)
3. **CLIP 토큰 제한**: 프롬프트 최대 77 토큰 (긴 설명은 자동 잘림)

### 알려진 이슈
1. **색상 정확도**: Stable Diffusion의 색상 재현이 100% 정확하지 않을 수 있음
   - 해결책: 프롬프트에 "EXACTLY {color} color" 명시, DPM Solver 스케줄러 사용
2. **YOLOv5 분류 정확도**: 반팔/긴팔 등 세부 분류가 부정확할 수 있음
   - 해결책: CLIP 후처리 검증으로 보완
---

## 🚧 향후 개선 사항

### 단기 목표
1. **YOLOv5 모델 학습 완료**: 100 epochs까지 학습하여 정확도 향상
2. **가상 피팅 속도 개선**: inference steps 최적화 (현재 11 steps for MPS, 7 steps for CPU)
3. **색상 정확도 향상**: ControlNet 또는 fine-tuning 적용
4. **코드 품질 개선**: ✅ 완료 - 중복 코드 제거, 공통 유틸리티 생성

### 중기 목표
1. **DeepFashion2 전체 데이터셋 학습**: 더 많은 클래스 지원
2. **사용자 피드백 기능**: 추천 결과 평가 및 학습
3. **SNS 트렌드 크롤링**: 실시간 트렌드 반영
4. **가상 피팅 품질 향상**: Segmentation 기반 정밀 마스킹

### 장기 목표
1. **개인화 모델**: 사용자별 선호도 학습
2. **전문 Virtual Try-On 모델**: VITON/CP-VTON 등 고품질 모델 통합
3. **쇼핑몰 연동**: 실제 제품 구매 연결

---

## 📚 참고 자료

### 데이터셋
- **DeepFashion2**: https://github.com/switchablenorms/DeepFashion2
- **ModaNet**: https://github.com/eBay/modanet

### 모델
- **YOLOv5**: https://github.com/ultralytics/yolov5
- **CLIP**: https://github.com/openai/CLIP
- **Stable Diffusion**: https://github.com/CompVis/stable-diffusion

### 문서
- **Streamlit**: https://docs.streamlit.io
- **Diffusers**: https://huggingface.co/docs/diffusers
- **MediaPipe**: https://google.github.io/mediapipe

---

## 👥 프로젝트 정보

**프로젝트명**: Fitzy  
**버전**: 1.0.0  
**개발 환경**: macOS, Python 3.12, Apple Silicon M2  
**라이선스**: MIT 
**마지막 업데이트**: 2025-11-10  

---

**문서 작성일**: 2025-11-10  
**작성자**: AI Assistant  
**문서 버전**: 1.1


