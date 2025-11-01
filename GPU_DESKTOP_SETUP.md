# GPU 데스크탑 환경 설정 및 학습 재개 가이드 🚀

## 📋 사전 준비 확인

**맥북에서 완료한 작업:**
- ✅ Git push 완료
- ✅ `deepfashion2_data.tar.gz` (1.2GB) 전송 완료
- ✅ `last.pt` (9.9MB) 전송 완료

**GPU 데스크탑에서 필요한 것:**
- ✅ Python 3.8 이상
- ✅ CUDA 지원 GPU (NVIDIA)
- ✅ 전송된 파일들 (압축 파일 + 체크포인트)

---

## Step 1: 프로젝트 클론

```bash
# 작업 디렉토리로 이동 (예: 홈 디렉토리 또는 프로젝트 폴더)
cd ~/projects  # 또는 원하는 경로

# Git 저장소 클론
git clone <your-repo-url> FItzy_copy
# 예: git clone https://github.com/username/FItzy_copy.git FItzy_copy

# 프로젝트 디렉토리로 이동
cd FItzy_copy
```

**확인:**
```bash
# 프로젝트 구조 확인
ls -la
# app.py, train_fashion.py, requirements.txt 등이 보여야 함
```

---

## Step 2: 가상환경 설정

```bash
# 가상환경 생성
python -m venv fitzy_env

# 가상환경 활성화
# Linux/Mac:
source fitzy_env/bin/activate

# Windows:
# fitzy_env\Scripts\activate

# 프롬프트에 (fitzy_env) 표시되면 성공
```

**확인:**
```bash
which python
# fitzy_env/bin/python 이 출력되어야 함
```

---

## Step 3: 패키지 설치

```bash
# pip 업그레이드
pip install --upgrade pip

# 필수 패키지 설치
pip install -r requirements.txt

# GPU 지원 확인 (선택)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**예상 설치 시간:** 5-10분 (인터넷 속도에 따라)

**GPU 확인 출력 예시:**
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3080
```

---

## Step 4: 데이터셋 압축 해제

**USB 드라이브가 마운트된 위치 확인:**
```bash
# Linux에서 USB 위치 확인
lsblk  # 또는
df -h | grep -i usb  # 또는
ls /mnt/  # 또는
ls /media/  # 또는
mount | grep -i usb
```

**일반적인 USB 마운트 위치:**
- Linux: `/mnt/usb` 또는 `/media/USB` 또는 `/media/username/USB_NAME`
- Windows: `D:\` 또는 `E:\` 등
- Mac: `/Volumes/USB_NAME`

```bash
# USB 드라이브에서 프로젝트 디렉토리로 파일 복사
# 예시 (경로는 실제 USB 마운트 위치로 변경):
cp /mnt/usb/deepfashion2_data.tar.gz ./
cp /mnt/usb/last.pt ./runs/train/yolov5_fashion2/weights/ 2>/dev/null || mkdir -p runs/train/yolov5_fashion2/weights/ && cp /mnt/usb/last.pt ./runs/train/yolov5_fashion2/weights/

# 또는 Windows의 경우:
# copy D:\deepfashion2_data.tar.gz .\
# mkdir runs\train\yolov5_fashion2\weights
# copy D:\last.pt runs\train\yolov5_fashion2\weights\

# 데이터셋 압축 해제
tar -xzf deepfashion2_data.tar.gz

# 확인
ls -lh deepfashion2_data/
# train/, valid/, test/, data.yaml 등이 보여야 함
```

**압축 해제 시간:** 약 1-2분

---

## Step 5: 체크포인트 복사

```bash
# 체크포인트 디렉토리 생성 (이미 있다면 무시됨)
mkdir -p runs/train/yolov5_fashion2/weights/

# USB에서 체크포인트 복사 (경로는 실제 USB 위치로 변경)
cp /mnt/usb/last.pt runs/train/yolov5_fashion2/weights/

# 또는 이미 Step 4에서 복사했다면 생략 가능

# 확인
ls -lh runs/train/yolov5_fashion2/weights/last.pt
# 9.9M 크기의 파일이 보여야 함
```

---

## Step 6: 학습 재개 실행 🎯

### 기본 실행 (권장)

```bash
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

### 옵션 설명

- `--resume`: 이어서 학습 모드
- `--resume-from`: 체크포인트 경로
- `--epochs 100`: 총 100 epochs (현재 1 완료, 99 남음)
- `--batch 32`: 배치 크기 (GPU 메모리에 따라 조정)
- `--device 0`: GPU 0번 사용 (GPU가 여러 개인 경우)

### GPU 메모리 부족 시

```bash
# 배치 크기 줄이기
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 16 \
  --device 0
```

### 더 빠른 학습을 원할 때

```bash
# 더 큰 모델 사용 (GPU 메모리 충분한 경우)
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --model s \
  --epochs 100 \
  --batch 32 \
  --device 0
```

---

## Step 7: 학습 모니터링

### 학습 진행 확인

학습 시작 시 다음과 같은 출력이 보입니다:

```
🔄 이어서 학습: runs/train/yolov5_fashion2/weights/last.pt
✅ 체크포인트 로드 완료
📌 이어서 학습 모드: 체크포인트에서 재개

Resuming training from runs/train/yolov5_fashion2/weights/last.pt

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100        2.5G      1.015      3.350      1.430         58        640
```

**정상 동작 확인:**
- ✅ "Resuming training" 메시지 표시
- ✅ Epoch 2부터 시작 (1이 완료되었으므로)
- ✅ GPU 메모리 사용량 표시
- ✅ Loss 값이 출력됨

### 학습 로그 확인

```bash
# 실시간 로그 확인
tail -f runs/train/yolov5_fashion2/results.csv

# 또는 학습 완료 후
cat runs/train/yolov5_fashion2/results.csv
```

### 예상 학습 시간

- **GPU (RTX 3080/3090 등)**: 약 5-8시간 (100 epochs)
- **GPU (RTX 3060 등)**: 약 8-12시간
- **Epoch당**: 약 3-5분

---

## 문제 해결 (Troubleshooting)

### 문제 1: CUDA 사용 불가

**증상:**
```
CUDA not available, using CPU
```

**해결:**
```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# CUDA 버전 확인
nvidia-smi

# CUDA 버전에 맞는 PyTorch 재설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 문제 2: 체크포인트를 찾을 수 없음

**증상:**
```
❌ 오류: 체크포인트를 찾을 수 없습니다
```

**해결:**
```bash
# 체크포인트 경로 확인
ls -lh runs/train/yolov5_fashion2/weights/last.pt

# 절대 경로로 지정
python train_fashion.py \
  --resume \
  --resume-from $(pwd)/runs/train/yolov5_fashion2/weights/last.pt \
  --device 0
```

### 문제 3: 데이터셋을 찾을 수 없음

**증상:**
```
❌ 오류: 데이터셋 설정 파일을 찾을 수 없습니다
```

**해결:**
```bash
# 데이터셋 확인
ls -la deepfashion2_data/
ls -la deepfashion2_data/data.yaml

# data.yaml 경로 확인
cat deepfashion2_data/data.yaml | head -5
```

### 문제 4: GPU 메모리 부족 (OOM)

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```bash
# 배치 크기 줄이기
python train_fashion.py ... --batch 16

# 또는 더 작은 모델 사용
python train_fashion.py ... --model n --batch 8
```

---

## 학습 완료 후

### 최종 모델 확인

```bash
# 최고 성능 모델
ls -lh runs/train/yolov5_fashion2/weights/best.pt

# 최종 모델
ls -lh runs/train/yolov5_fashion2/weights/last.pt

# 학습 결과
cat runs/train/yolov5_fashion2/results.csv | tail -5
```

### 모델을 앱에서 사용하기

```bash
# 학습된 모델을 앱 모델 디렉토리로 복사
mkdir -p models/weights/
cp runs/train/yolov5_fashion2/weights/best.pt models/weights/yolov5_fashion.pt

# 확인
ls -lh models/weights/yolov5_fashion.pt
```

---

## 전체 명령어 요약 (복사-붙여넣기용)

```bash
# 1. 프로젝트 클론
cd ~/projects
git clone <repo-url> FItzy_copy
cd FItzy_copy

# 2. 가상환경 설정
python -m venv fitzy_env
source fitzy_env/bin/activate  # Linux/Mac

# 3. 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. 데이터셋 및 체크포인트 복사 (USB 경로는 실제로 변경)
cp /mnt/usb/deepfashion2_data.tar.gz ./
cp /mnt/usb/last.pt runs/train/yolov5_fashion2/weights/ 2>/dev/null || mkdir -p runs/train/yolov5_fashion2/weights/ && cp /mnt/usb/last.pt runs/train/yolov5_fashion2/weights/

# 6. 데이터셋 압축 해제
tar -xzf deepfashion2_data.tar.gz

# 7. 파일 확인
ls -lh deepfashion2_data/data.yaml
ls -lh runs/train/yolov5_fashion2/weights/last.pt

# 8. 학습 재개
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

---

## 체크리스트 ✅

### 필수 항목
- [ ] 프로젝트 클론 완료
- [ ] 가상환경 생성 및 활성화
- [ ] 패키지 설치 완료
- [ ] CUDA 사용 가능 확인
- [ ] 데이터셋 압축 해제 완료
- [ ] 체크포인트 복사 완료
- [ ] 학습 재개 실행 성공

### 선택 항목
- [ ] 학습 로그 모니터링 설정
- [ ] 원격 접속 설정 (SSH 등)
- [ ] 학습 중단 후 재개 방법 확인

---

**이제 GPU 환경에서 빠르게 학습을 이어갈 수 있습니다!** 🚀

**예상 학습 시간:** 약 5-8시간 (GPU 기준)

