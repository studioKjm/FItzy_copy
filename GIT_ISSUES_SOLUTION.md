# Git 관리 문제 해결 가이드

## 문제 1: deepfashion2_data/가 소스 제어에 표시됨

### 원인 분석

`.gitignore`에 `deepfashion2_data/`를 추가했지만 여전히 표시되는 이유:

1. **이미 Git에 추적 중인 파일**: `.gitignore`는 **새로 추가되는 파일**만 무시합니다. 이미 `git add`로 추적 중인 파일은 무시되지 않습니다.
2. **Staged 상태**: 파일이 staging area에 있으면 표시됩니다.
3. **IDE 캐시**: IDE가 Git 상태를 캐시하고 있을 수 있습니다.

### 해결 방법

#### Step 1: 현재 Git 추적 상태 확인

```bash
# Git에 추적 중인 파일 확인
git ls-files | grep deepfashion2_data

# Staged 파일 확인
git status --short | grep deepfashion2_data
```

#### Step 2: Git 추적에서 제거 (파일은 유지)

**이미 추적 중인 경우:**
```bash
# Git 추적에서만 제거 (로컬 파일은 유지)
git rm -r --cached deepfashion2_data/

# 변경사항 커밋
git add .gitignore
git commit -m "Remove deepfashion2_data from Git tracking"
```

**⚠️ 주의**: `--cached` 옵션 없이 `git rm`을 사용하면 실제 파일이 삭제됩니다!

#### Step 3: IDE 새로고침

**VS Code / Cursor:**
- Git 패널에서 새로고침 버튼 클릭
- 또는 `Cmd+Shift+P` → "Reload Window"

### 완전한 해결 명령어

```bash
# 1. Git 추적에서 제거 (파일은 유지)
git rm -r --cached deepfashion2_data/

# 2. .gitignore 확인 (이미 추가되어 있어야 함)
grep "deepfashion2_data" .gitignore

# 3. 변경사항 커밋
git add .gitignore
git commit -m "Remove deepfashion2_data from Git tracking, keep in .gitignore"

# 4. 원격 저장소에 푸시
git push
```

**결과:**
- ✅ 로컬 파일은 유지됨
- ✅ Git 추적에서 제거됨
- ✅ 소스 제어 탭에 표시되지 않음
- ✅ 새로 추가되는 파일도 자동으로 무시됨

---

## 문제 2: Git 클론 후 이어서 학습 가능 여부

### ❌ 문제점

**`deepfashion2_data/`를 `.gitignore`에 추가하면:**

```
Git 저장소 클론
    ↓
deepfashion2_data/ 폴더 없음 ❌
    ↓
학습 불가능 ❌
```

**이유:**
- `.gitignore`된 파일/폴더는 Git에 포함되지 않음
- 클론 시 해당 폴더가 생성되지 않음
- 데이터셋이 없으면 학습 스크립트 실행 불가

### ✅ 해결 방법

#### 방법 A: 데이터셋 별도 전송 (권장)

**맥북에서:**
```bash
# 데이터셋 압축 (선택사항)
cd /Users/jimin/opensw/FItzy_copy
tar -czf deepfashion2_data.tar.gz deepfashion2_data/
```

**데스크탑에서:**
```bash
# 1. Git 클론 (코드만)
git clone <repo-url> FItzy_copy
cd FItzy_copy

# 2. 데이터셋 별도 전송
# USB 드라이브 사용:
tar -xzf /mnt/usb/deepfashion2_data.tar.gz

# 또는 직접 복사:
scp user@macbook:/Users/jimin/opensw/FItzy_copy/deepfashion2_data.tar.gz ./
tar -xzf deepfashion2_data.tar.gz

# 3. 이제 학습 가능
python train_fashion.py --resume --resume-from ...
```

#### 방법 B: README에 데이터셋 다운로드 안내 추가

**`DATASET_SETUP.md` 생성:**
```markdown
# 데이터셋 설정 가이드

## DeepFashion2 데이터셋 다운로드

Git 저장소에는 데이터셋이 포함되어 있지 않습니다.

### 다운로드 방법
1. [DeepFashion2 Small-32k 다운로드](링크)
2. `deepfashion2_data/` 폴더에 압축 해제
3. `deepfashion2_data/data.yaml` 경로 확인
```

#### 방법 C: Git LFS 사용 (제한적)

**주의:** Git LFS는 무료 플랜에서 **1GB 저장소, 1GB 대역폭/월**만 제공합니다.
- 데이터셋: 1.3GB → **무료 플랜 초과**

```bash
# Git LFS 설정
git lfs install
git lfs track "deepfashion2_data/**"

# ⚠️ 주의: GitHub 무료 플랜은 1GB만 제공
# 데이터셋 1.3GB는 초과함
```

**권장하지 않음**: 무료 플랜 제한 초과

---

## 권장 워크플로우

### 맥북 (개발 환경)

1. **Git 추적에서 데이터셋 제거**
   ```bash
   git rm -r --cached deepfashion2_data/
   git commit -m "Remove dataset from tracking"
   ```

2. **체크포인트는 별도 관리**
   - Git LFS 사용 (작은 파일)
   - 또는 USB/클라우드 사용

3. **코드만 Git에 푸시**
   ```bash
   git push
   ```

### GPU 데스크탑 (학습 환경)

1. **코드 클론**
   ```bash
   git clone <repo-url> FItzy_copy
   cd FItzy_copy
   ```

2. **데이터셋 별도 전송**
   ```bash
   # USB에서 복사
   cp -r /mnt/usb/deepfashion2_data ./
   
   # 또는 압축 해제
   tar -xzf deepfashion2_data.tar.gz
   ```

3. **체크포인트 복사**
   ```bash
   mkdir -p runs/train/yolov5_fashion2/weights/
   cp /mnt/usb/last.pt runs/train/yolov5_fashion2/weights/
   ```

4. **학습 실행**
   ```bash
   python train_fashion.py --resume --resume-from runs/train/yolov5_fashion2/weights/last.pt --device 0
   ```

---

## 체크리스트

### 맥북에서
- [ ] `git rm -r --cached deepfashion2_data/` 실행
- [ ] `.gitignore`에 `deepfashion2_data/` 확인
- [ ] 변경사항 커밋 및 푸시
- [ ] 데이터셋 압축 (선택): `tar -czf deepfashion2_data.tar.gz deepfashion2_data/`
- [ ] 체크포인트 백업: `runs/train/yolov5_fashion2/weights/last.pt`

### GPU 데스크탑에서
- [ ] `git clone`으로 코드 클론
- [ ] 데이터셋 전송 및 압축 해제
- [ ] 체크포인트 복사
- [ ] 학습 실행

---

## 요약

### ✅ Git 관리
- **코드**: Git에 포함
- **데이터셋**: `.gitignore`에 추가, Git 제외
- **체크포인트**: Git LFS 또는 수동 전송

### ✅ 이어서 학습 가능 여부
- **Git 클론만으로는 불가능** (데이터셋 없음)
- **데이터셋 별도 전송 필요**
- **전송 후에는 학습 가능** ✅

### ✅ 최종 답변

**질문: Git 클론만으로 이어서 학습 가능한가?**
- ❌ **불가능**: 데이터셋이 Git에 없음

**질문: 데이터셋 전송 후 이어서 학습 가능한가?**
- ✅ **가능**: 데이터셋 + 체크포인트 + 코드가 있으면 학습 가능

---

**이제 문제를 해결하세요!** 🚀

