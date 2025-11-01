# YOLOv5 패션 전용 모델 학습 가이드

## 개요

현재 Fitzy는 COCO 사전학습 모델(yolov5n)을 사용하여 `person`만 탐지합니다. 
패션 아이템 세부 탐지(상의, 하의, 가디건, 신발 등)를 위해서는 패션 전용 데이터셋으로 fine-tuning이 필요합니다.

## 필요한 데이터셋

### 1. DeepFashion2
- **설명**: 대규모 패션 데이터셋, 다양한 의류 카테고리
- **다운로드**: https://github.com/switchablenorms/DeepFashion2
- **카테고리**: 상의, 하의, 원피스, 외투, 가방, 신발 등

### 2. ModaNet
- **설명**: 패션 이미지 세그멘테이션 데이터셋
- **다운로드**: https://github.com/eBay/modanet
- **카테고리**: 상의, 하의, 드레스, 아우터, 신발 등

### 3. 커스텀 데이터셋
- 자체 수집한 패션 이미지 + 어노테이션

## 학습 단계

### 1. 데이터 준비

```bash
# 데이터셋 다운로드 및 구조화
data/
├── train/
│   ├── images/
│   └── labels/  # YOLO 형식 (.txt)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 2. 데이터 어노테이션

YOLO 형식 어노테이션 필요:
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4  # 클래스 0, 정규화된 좌표
```

**클래스 정의 예시:**
- 0: 상의 (top)
- 1: 하의 (bottom)
- 2: 원피스 (dress)
- 3: 가디건 (cardigan)
- 4: 코트 (coat)
- 5: 신발 (shoes)
- 6: 가방 (bag)

### 3. 학습 실행

```bash
# YOLOv5 학습 스크립트
from ultralytics import YOLO

# 사전학습 모델 로드
model = YOLO('yolov5n.pt')

# Fine-tuning 학습
results = model.train(
    data='fashion_dataset.yaml',  # 데이터셋 설정 파일
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov5_fashion'
)
```

### 4. 데이터셋 설정 파일 (fashion_dataset.yaml)

```yaml
path: ./data
train: train/images
val: val/images
test: test/images

nc: 7  # 클래스 수
names:
  0: top
  1: bottom
  2: dress
  3: cardigan
  4: coat
  5: shoes
  6: bag
```

### 5. 모델 저장 및 사용

학습 완료 후:
```python
# 모델 저장 경로
models/weights/yolov5_fashion.pt

# 코드에서 자동 로드됨
# src/models/models.py의 YOLODetector 클래스
```

## 예상 소요 시간

- **데이터 수집 및 어노테이션**: 1-2주 (수동 작업 시)
- **학습 시간**: 
  - GPU (RTX 3090): 6-12시간
  - CPU: 수일
- **검증 및 최적화**: 1주

## 권장 사양

- **GPU**: NVIDIA GPU (최소 8GB VRAM)
- **RAM**: 16GB 이상
- **저장공간**: 데이터셋 + 모델 (10GB 이상)

## 주의사항

1. **데이터 품질**: 다양한 각도, 조명, 스타일 포함
2. **클래스 불균형**: 모든 클래스가 충분히 포함되어야 함
3. **검증**: 학습 후 실제 이미지로 테스트 필수

## 참고 자료

- YOLOv5 공식 문서: https://docs.ultralytics.com/
- DeepFashion2: https://github.com/switchablenorms/DeepFashion2
- ModaNet: https://github.com/eBay/modanet

