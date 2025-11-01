# DeepFashion2 데이터셋 사용 가이드

## 1. 데이터셋 검토 결과

### ✅ 데이터셋 적절성: **양호**

**데이터셋 정보:**
- **이름**: DeepFashion2 Small-32k (Roboflow 버전)
- **크기**: 약 32,000개 이미지 (train: 10,346, valid: 995, test: 464)
- **형식**: YOLOv8 형식으로 이미 준비됨
- **클래스 수**: 13개 패션 아이템 카테고리

**클래스 목록 (13개):**
1. long sleeve dress (긴팔 드레스)
2. long sleeve outwear (긴팔 아우터)
3. long sleeve top (긴팔 상의)
4. short sleeve dress (반팔 드레스)
5. short sleeve outwear (반팔 아우터)
6. short sleeve top (반팔 상의)
7. shorts (반바지)
8. skirt (스커트)
9. sling dress (끈 드레스)
10. sling (끈 상의)
11. trousers (바지)
12. vest dress (조끼 드레스)
13. vest (조끼)

### 장점
✅ YOLO 형식으로 이미 준비됨 (변환 불필요)
✅ train/valid/test 분할 완료
✅ 이미지와 라벨 매칭 완료
✅ 패션 아이템 세부 분류 (상의/하의/드레스/아우터)
✅ 충분한 학습 데이터 (10,346개)

### 주의사항
⚠️ 원본 DeepFashion2의 일부만 포함 (Small 버전)
⚠️ 클래스 이름이 영어 (코드에서 한국어로 매핑 필요할 수 있음)

---

## 2. 폴더 이름 변경

**변경 완료**: `DeepFashion2 Small-32k.v1i.yolov8` → `deepfashion2_data`

```bash
# 변경 완료
/Users/jimin/opensw/FItzy_copy/deepfashion2_data/
```

---

## 3. 데이터셋 사용 단계별 가이드

### 3-1. 데이터셋 구조 확인

```
deepfashion2_data/
├── data.yaml          # YOLO 설정 파일
├── train/
│   ├── images/        # 10,346개 이미지
│   └── labels/        # 10,346개 라벨 (.txt)
├── valid/
│   ├── images/        # 995개 이미지
│   └── labels/        # 995개 라벨 (.txt)
└── test/
    ├── images/        # 464개 이미지
    └── labels/        # 464개 라벨 (.txt)
```

### 3-2. data.yaml 수정 완료 ✅

**수정 완료**: `deepfashion2_data/data.yaml` 경로 업데이트 완료

### 3-3. 학습 스크립트 생성 완료 ✅

#### 방법 1: 학습 스크립트 사용 (권장) ✅

**스크립트 위치**: `train_fashion.py` (프로젝트 루트)

**기본 실행:**
```bash
python train_fashion.py
```

**옵션 지정:**
```bash
# GPU 사용 시
python train_fashion.py --model s --epochs 100 --batch 32 --device 0

# CPU 사용 시 (느림)
python train_fashion.py --model n --epochs 50 --batch 4 --device cpu
```

#### 방법 2: Streamlit UI에서 학습 (향후 구현)

### 3-4. 학습 실행

```bash
# 가상환경 활성화
source fitzy_env/bin/activate

# 학습 스크립트 실행 (기본 설정)
python train_fashion.py

# 또는 GPU 사용 시
python train_fashion.py --model s --batch 32 --device 0
```

### 3-5. 학습된 모델 자동 배치

**자동 복사 완료** ✅
- 학습 스크립트가 자동으로 `best.pt`를 `models/weights/yolov5_fashion.pt`로 복사합니다.

**앱에서 자동 로드:**
- 다음 앱 실행 시 `src/models/models.py`의 `YOLODetector`가 자동으로 패션 전용 모델을 로드합니다.
- COCO 모델 대신 13개 패션 아이템을 탐지할 수 있습니다!

---

## 4. 학습 설정 권장값

### GPU 사용 시 (권장)
```python
model.train(
    data='deepfashion2_data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,      # GPU 메모리에 따라 16-64
    device=0,      # GPU 0번 사용
)
```

### CPU 사용 시 (느림)
```python
model.train(
    data='deepfashion2_data/data.yaml',
    epochs=50,     # CPU는 더 적게
    imgsz=640,
    batch=4,       # CPU는 작은 배치
    device='cpu',
)
```

---

## 5. 학습 모니터링

학습 중 다음 위치에서 결과 확인:
- **로그**: `runs/train/yolov5_fashion/`
- **모델 체크포인트**: `runs/train/yolov5_fashion/weights/`
  - `best.pt`: 최고 성능 모델
  - `last.pt`: 마지막 체크포인트

---

## 6. 다음 단계

1. ✅ **데이터셋 준비 완료** (deepfashion2_data/)
2. ✅ **폴더 이름 변경 완료** (deepfashion2_data/)
3. ✅ **data.yaml 경로 수정 완료**
4. ✅ **학습 스크립트 생성 완료** (`train_fashion.py`)
5. ⏳ **학습 실행** (GPU 권장) - 다음 단계!
6. ✅ **모델 자동 배치** (스크립트가 자동 처리)

---

## 7. 예상 소요 시간

- **GPU (RTX 3090)**: 약 2-4시간 (100 epochs)
- **GPU (RTX 3060)**: 약 4-6시간 (100 epochs)
- **CPU (M2 Mac)**: 약 1-2일 (50 epochs 권장)

---

## 8. 문제 해결

### 학습 중 메모리 부족
- `batch` 크기 줄이기 (16 → 8 → 4)
- `imgsz` 줄이기 (640 → 512)

### 학습이 너무 느림
- GPU 사용 확인: `torch.cuda.is_available()`
- 에폭 수 줄이기 (100 → 50)
- 배치 크기 늘리기 (GPU 메모리 허용 범위 내)

