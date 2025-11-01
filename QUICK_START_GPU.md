# GPU 데스크탑 빠른 시작 가이드 ⚡

## 한 번에 실행 (복사-붙여넣기)

```bash
# 1. 프로젝트 클론
cd ~/projects  # 또는 원하는 경로
git clone <your-repo-url> FItzy_copy
cd FItzy_copy

# 2. 가상환경 설정
python -m venv fitzy_env
source fitzy_env/bin/activate

# 3. 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. USB에서 파일 복사 (경로는 실제 USB 위치로 변경)
cp /mnt/usb/deepfashion2_data.tar.gz ./
mkdir -p runs/train/yolov5_fashion2/weights/
cp /mnt/usb/last.pt runs/train/yolov5_fashion2/weights/

# 5. 데이터셋 압축 해제
tar -xzf deepfashion2_data.tar.gz

# 6. 학습 재개
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

## USB 경로 찾기

**Linux:**
```bash
# USB 마운트 위치 확인
lsblk
# 또는
df -h | grep -i usb
# 또는
ls /media/$(whoami)/  # 일반적인 위치
```

**Windows:**
```
D:\ 또는 E:\ 등 (탐색기에서 확인)
```

**경로 예시:**
- `/mnt/usb/`
- `/media/username/USB_NAME/`
- `/media/usb/`

---

## 필수 확인 사항

### 1. GPU 확인
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# CUDA: True 가 출력되어야 함
```

### 2. 파일 확인
```bash
# 데이터셋
ls -lh deepfashion2_data/data.yaml

# 체크포인트
ls -lh runs/train/yolov5_fashion2/weights/last.pt
```

### 3. 학습 시작 확인
```
🔄 이어서 학습: ...
✅ 체크포인트 로드 완료
📌 이어서 학습 모드: 체크포인트에서 재개
Resuming training from ...
Epoch 2/100: ...
```

---

## 문제 발생 시

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python train_fashion.py ... --batch 16
```

### 파일 경로 오류
```bash
# 절대 경로 사용
python train_fashion.py \
  --resume-from $(pwd)/runs/train/yolov5_fashion2/weights/last.pt \
  ...
```

---

**상세 가이드:** `GPU_DESKTOP_SETUP.md` 참조

