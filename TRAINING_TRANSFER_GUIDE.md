# GPU 환경으로 학습 전환 가이드

## 📋 목차
1. [현재 학습 상태 브리핑](#1-현재-학습-상태-브리핑)
2. [안전한 학습 중단 방법](#2-안전한-학습-중단-방법)
3. [Git 관리 방법 (대용량 파일)](#3-git-관리-방법-대용량-파일)
4. [GPU 환경에서 이어서 학습](#4-gpu-환경에서-이어서-학습)

---

## 1. 현재 학습 상태 브리핑

### ✅ 완료된 학습 상태

**학습 진행 상황:**
- **Epoch**: 1/100 완료 (1%)
- **학습 시간**: 약 57분 (3414.88초)
- **체크포인트**: ✅ 저장 완료
  - `runs/train/yolov5_fashion2/weights/last.pt` (9.9MB)
  - `runs/train/yolov5_fashion2/weights/best.pt` (9.9MB)

**학습 성능 지표 (Epoch 1 완료):**
- **Train Loss**: 
  - box_loss: 1.01892
  - cls_loss: 3.37446
  - dfl_loss: 1.43737
- **Validation Loss**: 
  - box_loss: 1.04766
  - cls_loss: 3.22427
  - dfl_loss: 1.54744
- **mAP50**: 0.13955 (13.9%)
- **mAP50-95**: 0.09123 (9.1%)
- **Precision**: 0.27203 (27.2%)
- **Recall**: 0.19584 (19.6%)

**Loss 감소 추이:**
- 초기 (Batch 1): box=1.516, cls=4.181, dfl=1.87
- 완료 (Epoch 1): box=1.019, cls=3.374, dfl=1.437
- **33-20% 감소** ✅ 정상 학습 진행 중

### 📁 저장된 파일 위치

```
runs/train/yolov5_fashion2/
├── weights/
│   ├── best.pt      # 최고 성능 모델 (9.9MB)
│   └── last.pt      # 마지막 체크포인트 (9.9MB) ⭐ 이어서 학습에 사용
├── results.csv      # 학습 로그
├── args.yaml        # 학습 설정
└── train_batch*.jpg # 학습 시각화

deepfashion2_data/
├── data.yaml        # 데이터셋 설정
├── train/           # 학습 데이터 (10,346 이미지)
└── valid/           # 검증 데이터 (995 이미지)
```

---

## 2. 안전한 학습 중단 방법

### ✅ YOLO는 자동으로 체크포인트 저장

YOLO는 학습 중 **자동으로 체크포인트를 저장**하므로 안전하게 중단 가능합니다.

### 방법: Ctrl+C로 정상 종료

```bash
# 학습 중인 터미널에서
Ctrl + C

# YOLO가 현재 배치를 완료한 후 안전하게 종료합니다
# last.pt와 best.pt가 최신 상태로 저장됩니다
```

**⚠️ 주의사항:**
- ✅ **Ctrl+C는 안전함**: YOLO가 배치 완료 후 체크포인트 저장 후 종료
- ❌ **터미널 강제 종료는 비추천**: 마지막 배치 손실 가능
- ✅ **체크포인트는 Epoch마다 저장**: Epoch 1 완료 상태이므로 이미 저장됨

### ✅ 현재 상태 확인

학습이 이미 Epoch 1을 완료했으므로:
- ✅ `last.pt`: Epoch 1 완료 상태 저장됨
- ✅ `best.pt`: 현재까지 최고 성능 모델 저장됨
- ✅ `results.csv`: 학습 로그 기록됨

**→ 안전하게 중단 가능합니다!**

---

## 3. Git 관리 방법 (대용량 파일)

### 문제점
- 모델 파일: `*.pt` (약 10MB)
- 학습 결과: `runs/train/` (수백 MB)
- 데이터셋: `deepfashion2_data/` (수 GB)

**Git은 일반적으로 100MB 이상 파일 관리에 부적합**

### 해결 방법: Git LFS (Large File Storage)

#### 3-1. Git LFS 설치

**macOS:**
```bash
brew install git-lfs
```

**Linux:**
```bash
sudo apt install git-lfs  # Ubuntu/Debian
```

**Windows:**
```bash
# Git for Windows에 포함되어 있음
# 또는 https://git-lfs.github.com 에서 다운로드
```

#### 3-2. Git LFS 초기화

```bash
cd /Users/jimin/opensw/FItzy_copy

# Git LFS 초기화
git lfs install

# .pt 파일을 LFS로 추적
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.ckpt"

# .gitattributes 파일 커밋
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
```

#### 3-3. .gitignore 설정 (대용량 파일 제외)

**권장 방식: 대용량 파일은 Git에 포함하지 않음**

`.gitignore`에 추가:
```gitignore
# 대용량 학습 결과 (로컬/원격 환경에서 별도 관리)
runs/
*.pt
*.pth
*.ckpt

# 데이터셋 (각 환경에서 별도 다운로드)
deepfashion2_data/

# 예외: 중요한 체크포인트만 LFS로 관리 (선택사항)
# !runs/train/yolov5_fashion2/weights/last.pt
```

#### 3-4. 권장 Git 관리 전략

**옵션 A: 체크포인트만 LFS로 관리 (권장)**

```bash
# .gitignore에 runs/ 추가 후
# 중요 체크포인트만 LFS로 추적
git lfs track "runs/train/*/weights/last.pt"
git lfs track "runs/train/*/weights/best.pt"

git add .gitattributes
git add runs/train/yolov5_fashion2/weights/last.pt
git commit -m "Add checkpoint to Git LFS"
git push
```

**옵션 B: 모든 대용량 파일 제외 (더 간단)**

```bash
# .gitignore에 모든 대용량 파일 추가
# Git에는 코드와 설정만 포함
# 체크포인트는 수동으로 전송 (USB, 클라우드 등)
```

### 3-5. 체크포인트 수동 전송 방법

**방법 1: USB 드라이브**
```bash
# 맥북에서
cp runs/train/yolov5_fashion2/weights/last.pt /Volumes/USB/

# 데스크탑에서
# USB 마운트 후 복사
```

**방법 2: 클라우드 스토리지**
```bash
# Google Drive, Dropbox, OneDrive 등 사용
# 또는 rsync로 직접 전송
rsync -avz runs/train/yolov5_fashion2/weights/last.pt user@desktop:/path/to/project/
```

**방법 3: GitHub Releases (제한적)**
```bash
# 큰 파일이지만 100MB 미만이면 Releases에 업로드 가능
# 100MB 이상은 Git LFS 필요
```

---

## 4. GPU 환경에서 이어서 학습

### 4-1. 파일 준비 (GPU 데스크탑)

**필요한 파일:**
1. ✅ **체크포인트**: `runs/train/yolov5_fashion2/weights/last.pt` ⭐
2. ✅ **데이터셋**: `deepfashion2_data/` 전체 폴더
3. ✅ **학습 스크립트**: `train_fashion.py`
4. ✅ **설정 파일**: `deepfashion2_data/data.yaml`

### 4-2. GPU 환경 설정

```bash
# 1. 프로젝트 클론/복사
cd /path/to/desktop/project
git clone <your-repo-url> FItzy_copy  # 또는 rsync로 복사

# 2. 가상환경 생성 및 패키지 설치
python -m venv fitzy_env
source fitzy_env/bin/activate  # Linux/Mac
# 또는 fitzy_env\Scripts\activate  # Windows

pip install -r requirements.txt

# 3. 데이터셋 복사 (수동으로 전송)
# deepfashion2_data/ 폴더 전체를 프로젝트 루트에 복사

# 4. 체크포인트 복사
mkdir -p runs/train/yolov5_fashion2/weights/
# last.pt를 위 경로에 복사
```

### 4-3. 이어서 학습 (Resume Training)

**YOLO는 `resume` 옵션으로 이어서 학습 가능**

#### 방법 1: train_fashion.py 수정 (권장)

`train_fashion.py`에 resume 기능 추가:

```python
def train_fashion_model(
    model_size="n",
    epochs=100,
    batch_size=16,
    img_size=640,
    device="cpu",
    resume=False,  # 추가
    resume_from=None  # 추가: 체크포인트 경로
):
    # ...
    
    try:
        original_dir = os.getcwd()
        os.chdir(DATA_DIR)
        
        try:
            # Resume 옵션 추가
            train_args = {
                'data': str(DATA_YAML.name),
                'epochs': epochs,
                'imgsz': img_size,
                'batch': batch_size,
                'name': 'yolov5_fashion',
                'project': str(BASE_DIR / 'runs' / 'train'),
                'patience': 50,
                'save': True,
                'val': True,
                'device': device,
                'workers': 4 if device != "cpu" else 0,
            }
            
            # Resume 학습
            if resume and resume_from:
                train_args['resume'] = True
                # resume_from은 전체 경로 또는 상대 경로
                results = model.train(**train_args, resume=str(resume_from))
            else:
                results = model.train(**train_args)
                
        finally:
            os.chdir(original_dir)
```

#### 방법 2: 직접 Ultralytics YOLO 사용

```python
from ultralytics import YOLO

# 체크포인트에서 모델 로드
model = YOLO('runs/train/yolov5_fashion2/weights/last.pt')

# 이어서 학습
results = model.train(
    resume=True,  # 이어서 학습
    epochs=100,   # 총 epochs (현재 1 완료, 99 남음)
    imgsz=640,
    batch=32,     # GPU는 더 큰 배치 가능
    device=0,     # GPU 0번 사용
)
```

#### 방법 3: 명령줄에서 직접 실행

```bash
cd /path/to/project

# 체크포인트 경로 지정하여 이어서 학습
python -c "
from ultralytics import YOLO
import os
os.chdir('deepfashion2_data')
model = YOLO('../runs/train/yolov5_fashion2/weights/last.pt')
model.train(
    data='data.yaml',
    resume=True,
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    name='yolov5_fashion',
    project='../runs/train'
)
"
```

### 4-4. 학습 재개 확인

**학습이 정상적으로 재개되면:**
```
Resuming training from runs/train/yolov5_fashion2/weights/last.pt
Epoch 2/100: ...
```

**체크포인트 정보 자동 로드:**
- 이전 epoch 정보
- 최적화기 상태
- 학습률 스케줄
- 모델 가중치

---

## 5. 단계별 실행 가이드

### Step 1: 맥북에서 안전한 중단

```bash
# 1. 현재 학습 확인
ps aux | grep train_fashion

# 2. 안전하게 중단 (Ctrl+C)
# 이미 Epoch 1 완료 상태이므로 안전함

# 3. 체크포인트 확인
ls -lh runs/train/yolov5_fashion2/weights/
```

### Step 2: 필수 파일 확인

```bash
# 체크포인트 (필수)
runs/train/yolov5_fashion2/weights/last.pt  # 9.9MB

# 데이터셋 (필수)
deepfashion2_data/data.yaml
deepfashion2_data/train/
deepfashion2_data/valid/

# 학습 스크립트 (선택, Git에 있으면 클론 가능)
train_fashion.py
```

### Step 3: 파일 전송

**옵션 A: USB 드라이브**
```bash
# 맥북
cp -r runs/train/yolov5_fashion2/weights/ /Volumes/USB/checkpoint/
cp -r deepfashion2_data/ /Volumes/USB/

# 데스크탑
cp /mnt/usb/checkpoint/last.pt runs/train/yolov5_fashion2/weights/
cp -r /mnt/usb/deepfashion2_data/ ./
```

**옵션 B: Git (LFS 사용)**
```bash
# 맥북
git lfs track "runs/train/*/weights/*.pt"
git add runs/train/yolov5_fashion2/weights/last.pt
git commit -m "Add checkpoint for resume"
git push

# 데스크탑
git pull
git lfs pull  # LFS 파일 다운로드
```

### Step 4: GPU 환경에서 재개

```bash
# 데스크탑에서
cd /path/to/FItzy_copy
source fitzy_env/bin/activate

# 방법 1: Python 스크립트
python -c "
from ultralytics import YOLO
import os
os.chdir('deepfashion2_data')
model = YOLO('../runs/train/yolov5_fashion2/weights/last.pt')
model.train(
    data='data.yaml',
    resume=True,
    epochs=100,
    batch=32,
    device=0,
    name='yolov5_fashion',
    project='../runs/train'
)
"

# 방법 2: train_fashion.py 사용 (resume 기능 추가 필요)
```

### Step 5: 학습 완료 후 맥북으로 가져오기

```bash
# GPU 환경에서 학습 완료 후
# 최종 모델 복사
cp runs/train/yolov5_fashion2/weights/best.pt models/weights/yolov5_fashion.pt

# 맥북으로 전송 (USB 또는 Git)
```

---

## 6. 요약 및 체크리스트

### ✅ 맥북에서 할 일

- [ ] 학습 안전하게 중단 (Ctrl+C)
- [ ] 체크포인트 확인: `last.pt`, `best.pt`
- [ ] 필수 파일 목록 작성
- [ ] 파일 전송 준비 (USB 또는 Git LFS)

### ✅ GPU 데스크탑에서 할 일

- [ ] 프로젝트 복사/클론
- [ ] 가상환경 설정 및 패키지 설치
- [ ] 데이터셋 복사 (`deepfashion2_data/`)
- [ ] 체크포인트 복사 (`last.pt`)
- [ ] 이어서 학습 실행 (`resume=True`)
- [ ] 학습 완료 후 최종 모델 확인

### ✅ 주의사항

1. **체크포인트 경로**: 상대 경로 주의 (프로젝트 구조 동일하게 유지)
2. **데이터셋 경로**: `data.yaml`의 경로 확인
3. **Python 버전**: 가능하면 동일 버전 사용
4. **PyTorch/CUDA**: GPU 환경에 맞는 버전 설치

---

## 7. 문제 해결

### 문제: "Cannot find checkpoint"
- 체크포인트 파일 경로 확인
- 절대 경로로 지정해보기

### 문제: "Dataset not found"
- `deepfashion2_data/` 폴더 위치 확인
- `data.yaml`의 경로 설정 확인

### 문제: "Resume failed"
- `last.pt` 파일 무결성 확인
- Epoch 1 완료 상태인지 확인
- 처음부터 다시 학습: `resume=False`로 시작

---

**이 가이드를 따라하면 안전하게 GPU 환경으로 전환하여 학습을 이어갈 수 있습니다!** 🚀

