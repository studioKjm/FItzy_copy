# Git 문제 즉시 해결 가이드 🛠️

## 문제 1: deepfashion2_data/ 파일들이 소스 제어에 표시됨

### 현재 상태 확인

`.gitignore`에 추가했지만 여전히 표시되는 이유:
- IDE 캐시 문제일 가능성 높음
- Git에 추적 중이 아닐 수도 있음 (untracked files로 표시)

### 즉시 해결 방법

```bash
# 1. Git 추적 상태 확인
git ls-files | grep deepfashion2_data

# 결과가 없으면 = Git에 추적되지 않음 (정상)
# 결과가 있으면 = 아래 명령어 실행
```

**만약 Git에 추적 중이라면:**
```bash
# Git 추적에서 제거 (파일은 유지)
git rm -r --cached deepfashion2_data/

# 커밋
git add .gitignore
git commit -m "Remove deepfashion2_data from Git tracking"
```

**IDE 새로고침:**
- VS Code/Cursor: `Cmd+Shift+P` → "Reload Window"
- 또는 Git 패널 새로고침

---

## 문제 2: Git 클론 후 학습 가능 여부

### ❌ 답: 클론만으로는 불가능

**이유:**
```
Git 클론
  ↓
deepfashion2_data/ 없음 (gitignore로 제외됨)
  ↓
학습 스크립트 실행 불가 ❌
```

### ✅ 해결: 데이터셋 별도 전송 필요

**필수 파일:**
1. ✅ 코드 (Git에 포함)
2. ✅ 데이터셋 `deepfashion2_data/` (1.3GB) - 별도 전송 필요
3. ✅ 체크포인트 `last.pt` (9.9MB) - 별도 전송 필요

---

## 권장 워크플로우

### Step 1: 맥북에서 준비

```bash
# 1. Git 추적 정리 (필요시)
git rm -r --cached deepfashion2_data/ 2>/dev/null || true

# 2. 데이터셋 압축 (전송 용이)
cd /Users/jimin/opensw/FItzy_copy
tar -czf deepfashion2_data.tar.gz deepfashion2_data/

# 3. 체크포인트 확인
ls -lh runs/train/yolov5_fashion2/weights/last.pt
```

### Step 2: 파일 전송

**옵션 A: USB 드라이브 (권장)**
```bash
# 맥북
cp deepfashion2_data.tar.gz /Volumes/USB/
cp runs/train/yolov5_fashion2/weights/last.pt /Volumes/USB/
```

**옵션 B: rsync (네트워크)**
```bash
rsync -avz deepfashion2_data.tar.gz user@desktop:/path/to/project/
rsync -avz runs/train/yolov5_fashion2/weights/last.pt user@desktop:/path/to/project/
```

### Step 3: GPU 데스크탑에서 설정

```bash
# 1. 코드 클론
git clone <repo-url> FItzy_copy
cd FItzy_copy

# 2. 가상환경 설정
python -m venv fitzy_env
source fitzy_env/bin/activate
pip install -r requirements.txt

# 3. 데이터셋 압축 해제
tar -xzf /mnt/usb/deepfashion2_data.tar.gz

# 4. 체크포인트 복사
mkdir -p runs/train/yolov5_fashion2/weights/
cp /mnt/usb/last.pt runs/train/yolov5_fashion2/weights/

# 5. 학습 실행 ✅
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolov5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

---

## 최종 답변

### Q1: Git 클론만으로 이어서 학습 가능한가?
**❌ 아니요.** 데이터셋이 Git에 없으므로 불가능합니다.

### Q2: 데이터셋을 별도로 전송하면 가능한가?
**✅ 네.** 데이터셋 + 체크포인트 + 코드가 있으면 가능합니다.

### Q3: IDE에서 파일들이 계속 표시되는 이유는?
- `.gitignore`가 적용되었지만 IDE 캐시 문제
- 또는 Untracked files로 표시 중 (정상)
- IDE 새로고침으로 해결 가능

---

**자세한 내용은 `GIT_ISSUES_SOLUTION.md` 참조**

