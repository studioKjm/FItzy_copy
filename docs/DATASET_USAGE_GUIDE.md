# DeepFashion2 데이터셋 사용 가이드

## 1. 데이터셋 검토 결과
 
**데이터셋 정보:**
- **이름**: DeepFashion2 Small-32k (Roboflow 버전)
- **크기**: 약 11,000개 이미지 (train: 10,346, valid: 995, test: 464)
- **형식**: YOLOv8 형식
- **클래스 수**: 13개 패션 아이템 카테고리

**클래스 목록 (13개):**

현재 구현된 클래스 매핑 (`src/models/models.py`의 `YOLODetector.FASHION_CLASS_MAPPING`):

```python
FASHION_CLASS_MAPPING = {
    "long sleeve dress": "긴팔 드레스",
    "long sleeve outwear": "긴팔 아우터",
    "long sleeve top": "긴팔 상의",
    "short sleeve dress": "반팔 드레스",
    "short sleeve outwear": "반팔 아우터",
    "short sleeve top": "반팔 상의",
    "shorts": "반바지",
    "skirt": "스커트",
    "sling dress": "끈 드레스",
    "sling": "끈 상의",
    "trousers": "바지",
    "vest dress": "조끼 드레스",
    "vest": "조끼"
}
```

**클래스 분류:**
- **상의**: long/short sleeve top, sling, vest (5개)
- **하의**: shorts, skirt, trousers (3개)
- **드레스**: long/short sleeve dress, sling dress, vest dress (5개)
- **아우터**: long/short sleeve outwear (2개)

**데이터셋 특징:**
- YOLO 형식으로 이미 준비됨 (변환 불필요)
- train/valid/test 분할 완료
- 이미지와 라벨 매칭 완료
- 패션 아이템 세부 분류 (상의/하의/드레스/아우터)
- 충분한 학습 데이터 (train: 10,346개)

**데이터셋 한계:**
- 원본 DeepFashion2의 일부만 포함 (Small 버전)
- **여성 의상 중심 구성**: `dress`, `skirt`, `sling` 등 여성 의상 클래스가 다수 포함되어 있어, 데이터셋 특성상 여성 의상에 대한 탐지 정확도가 더 높을 수 있음
- 남성 의상 탐지 시 `trousers`, `shorts`, `top`, `outwear` 클래스는 정상 작동하지만, 학습 데이터가 상대적으로 적어 정확도가 낮을 수 있음
- 클래스 이름이 영어로 저장되며, 코드에서 `FASHION_CLASS_MAPPING`을 통해 한국어로 변환
- 실제 탐지 결과는 학습된 모델의 성능과 입력 이미지에 따라 달라짐


---



## 2. 데이터셋 경로

데이터셋은 프로젝트 루트의 `deepfashion2_data/` 디렉터리에 위치해야 합니다.

```bash
/~FItzy/deepfashion2_data/
```

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

### 3-2. data.yaml 설정

`deepfashion2_data/data.yaml` 파일에서 데이터셋 경로 올바르게 설정 필요

### 3-3. 학습 스크립트

**스크립트 위치**: `train_fashion.py` (프로젝트 루트)

학습 스크립트는 Ultralytics YOLO를 사용하여 모델을 학습합니다.

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

### 3-4. 학습 실행

```bash
# 가상환경 활성화
source fitzy_env/bin/activate

# 학습 스크립트 실행 (기본 설정)
python train_fashion.py

# 또는 GPU 사용 시
python train_fashion.py --model s --batch 32 --device 0
```

### 3-5. 학습된 모델 사용

**모델 저장 위치:**
- 학습 완료 후 `runs/train/yolov5_fashion/weights/best.pt`에 최고 성능 모델 저장
- 학습 스크립트가 자동으로 `best.pt`를 `models/weights/yolov5_fashion.pt`로 복사

**앱에서 모델 로드:**
- `src/models/models.py`의 `YOLODetector` 클래스는 `config.py`의 `YOLO_MODEL_PATH`에서 모델 경로를 읽음
- 기본 경로: `models/weights/yolov5_fashion.pt` (`config.py`에서 정의)
- 모델 파일이 존재하면 패션 전용 모델을 로드하고, 없으면 COCO 사전 학습 모델(`yolov5n.pt`)을 사용
- 패션 모델 로드 시 `is_fashion_model = True`로 설정되어 13개 패션 클래스를 탐지
- 탐지 시 신뢰도 임계값 0.3 이상인 경우만 반환 (`detect_clothes` 메서드)
- CLIP 검증을 통해 긴팔/반팔 등 세부 분류 정확도 향상 (`_verify_detection_with_clip` 메서드)

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

## 6. 모델 학습 및 사용 흐름

1. **데이터셋 준비**: `deepfashion2_data/` 디렉터리에 데이터셋 배치
2. **data.yaml 확인**: 데이터셋 경로가 올바르게 설정되어 있는지 확인
3. **학습 실행**: `train_fashion.py` 스크립트 실행 (GPU 권장)
4. **모델 자동 배치**: 학습 완료 후 `best.pt`가 `models/weights/yolov5_fashion.pt`로 자동 복사
5. **앱에서 사용**: 다음 앱 실행 시 `YOLODetector`가 패션 전용 모델을 자동 로드
6. **탐지 결과**: 13개 패션 클래스를 탐지하며, `FASHION_CLASS_MAPPING`을 통해 한국어로 변환되어 표시

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

