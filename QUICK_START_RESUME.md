# 빠른 시작: GPU 환경에서 이어서 학습하기 🚀

## 간단 요약

1. **맥북**: 학습 중단 (Ctrl+C) - 이미 Epoch 1 완료, 안전함 ✅
2. **파일 전송**: `last.pt` (9.9MB) + `deepfashion2_data/` (1.3GB)
3. **GPU 데스크탑**: 아래 명령어 실행

```bash
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

---

## 상세 가이드

### 1. 맥북에서 안전한 중단

```bash
# 터미널에서 Ctrl+C
# 이미 Epoch 1 완료 상태이므로 안전함
```

### 2. 필수 파일 확인

```bash
# 체크포인트 확인
ls -lh runs/train/yolov5_fashion2/weights/last.pt  # 9.9MB

# 데이터셋 확인
ls -lh deepfashion2_data/  # 약 1.3GB
```

### 3. 파일 전송

**USB 드라이브 사용 (권장):**
```bash
# 맥북
cp runs/train/yolov5_fashion2/weights/last.pt /Volumes/USB/
cp -r deepfashion2_data /Volumes/USB/
```

**Git LFS 사용 (선택):**
```bash
# Git LFS 설정 (최초 1회)
git lfs install
git lfs track "runs/train/*/weights/*.pt"

# 커밋 및 푸시
git add .gitattributes
git add runs/train/yolov5_fashion2/weights/last.pt
git commit -m "Add checkpoint for resume training"
git push
```

### 4. GPU 데스크탑에서 설정

```bash
# 1. 프로젝트 클론 (또는 USB에서 복사)
cd /path/to/project

# 2. 가상환경 설정
python -m venv fitzy_env
source fitzy_env/bin/activate  # Linux/Mac
# 또는 fitzy_env\Scripts\activate  # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 데이터셋 및 체크포인트 복사
# USB에서 또는 Git LFS pull
```

### 5. 이어서 학습 실행

```bash
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

**예상 시간**: 약 5-8시간 (100 epochs, GPU 기준)

---

## 자세한 가이드

- **전체 가이드**: `TRAINING_TRANSFER_GUIDE.md`
- **현재 상태**: `CURRENT_TRAINING_STATUS.md`

