# 학습 결과 맥북 전송 가이드

## 📊 전송할 파일 요약

**위치**: `runs/train/yolo5_fashion2/`  
**총 크기**: 약 20MB  
**Epoch 완료**: 29개 (Epoch 2-30)

### 필수 파일 목록:
1. ✅ `weights/best.pt` (9.91MB) - 최고 성능 모델
2. ✅ `weights/last.pt` (9.91MB) - 마지막 체크포인트 (이어서 학습용)
3. ✅ `results.csv` - 학습 로그 (29 epochs)
4. ✅ `args.yaml` - 학습 설정 파일
5. ⚠️ `labels.jpg` - 시각화 (선택사항)

---

## 방법 1: USB 드라이브 사용 (권장) 🚀

### Windows 데스크탑에서:

```powershell
# 1. USB 드라이브 확인 (예: E: 드라이브)
Get-Volume | Where-Object {$_.DriveType -eq 'Removable'}

# 2. 학습 결과 폴더를 USB로 복사
Copy-Item -Recurse runs\train\yolo5_fashion2 E:\training_results\

# 또는 압축해서 전송 (권장)
Compress-Archive -Path runs\train\yolo5_fashion2 -DestinationPath E:\training_results_yolo5_fashion2.zip
```

### 맥북에서:

```bash
# 1. USB 마운트 확인
ls /Volumes/

# 2. 압축 해제 후 프로젝트로 복사
unzip /Volumes/USB/training_results_yolo5_fashion2.zip -d /tmp/
cp -r /tmp/runs/train/yolo5_fashion2 /Users/jimin/opensw/FItzy_copy/runs/train/

# 또는 직접 복사 (압축 안 한 경우)
cp -r /Volumes/USB/training_results/yolo5_fashion2 /Users/jimin/opensw/FItzy_copy/runs/train/
```

---

## 방법 2: 네트워크 전송 (같은 네트워크에 연결된 경우)

### Windows에서 공유 폴더 생성:

```powershell
# 공유 폴더 생성 (예: C:\Shared)
New-Item -ItemType Directory -Path "C:\Shared" -Force
Copy-Item -Recurse runs\train\yolo5_fashion2 C:\Shared\

# 공유 설정 (관리자 권한 필요)
net share TrainingResults=C:\Shared /grant:Everyone,Full
```

### 맥북에서 접근:

```bash
# Finder에서: Cmd+K → smb://[Windows_IP_ADDRESS]/TrainingResults
# 또는 터미널에서:
mkdir -p ~/Desktop/training_results
cp -r /Volumes/TrainingResults/yolo5_fashion2 ~/Desktop/training_results/
```

---

## 방법 3: Git LFS 사용 (Git 저장소 사용 중인 경우)

### 주의: `runs/` 폴더는 `.gitignore`에 포함되어 있음

#### 옵션 A: 임시로 .gitignore 수정하여 커밋

**Windows 데스크탑에서:**

```powershell
# .gitignore에서 runs/ 주석 처리
(Get-Content .gitignore) -replace '^runs/$', '# runs/' | Set-Content .gitignore

# Git LFS 설정 (최초 1회)
git lfs install
git lfs track "runs/train/*/weights/*.pt"

# 파일 추가 및 커밋
git add .gitattributes
git add runs/train/yolo5_fashion2/
git commit -m "Add training results from GPU desktop (30 epochs)"
git push

# .gitignore 복구
(Get-Content .gitignore) -replace '^# runs/$', 'runs/' | Set-Content .gitignore
```

**맥북에서:**

```bash
git pull
git lfs pull  # LFS 파일 다운로드
```

#### 옵션 B: Git LFS 없이 직접 커밋 (20MB는 LFS 없이도 가능)

```powershell
# .gitignore 임시 수정
(Get-Content .gitignore) -replace '^runs/$', '# runs/' | Set-Content .gitignore

git add runs/train/yolo5_fashion2/
git commit -m "Add training results (30 epochs)"
git push

# .gitignore 복구
(Get-Content .gitignore) -replace '^# runs/$', 'runs/' | Set-Content .gitignore
```

---

## 방법 4: 클라우드 스토리지 (Google Drive, Dropbox 등)

### Windows에서:

```powershell
# Google Drive/Dropbox 폴더로 복사
Copy-Item -Recurse runs\train\yolo5_fashion2 "$env:USERPROFILE\Google Drive\TrainingResults\"
```

### 맥북에서:

```bash
# 클라우드 폴더에서 프로젝트로 복사
cp -r ~/Google\ Drive/TrainingResults/yolo5_fashion2 /Users/jimin/opensw/FItzy_copy/runs/train/
```

---

## 방법 5: 압축 파일로 전송

### Windows에서 압축:

```powershell
# ZIP 파일 생성
Compress-Archive -Path runs\train\yolo5_fashion2 -DestinationPath training_results_30epochs.zip -Force

# 파일 크기 확인
Get-Item training_results_30epochs.zip | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB, 2)}}
```

그 후 USB, 이메일, 클라우드 등을 통해 전송

### 맥북에서 압축 해제:

```bash
# 압축 해제
unzip training_results_30epochs.zip -d /tmp/

# 프로젝트로 이동
cp -r /tmp/runs/train/yolo5_fashion2 /Users/jimin/opensw/FItzy_copy/runs/train/
```

---

## ✅ 전송 완료 후 확인

맥북에서 다음 명령어로 확인:

```bash
# 파일 확인
ls -lh runs/train/yolo5_fashion2/weights/

# 학습 결과 확인
head -5 runs/train/yolo5_fashion2/results.csv
tail -3 runs/train/yolo5_fashion2/results.csv

# 총 epoch 수 확인
wc -l runs/train/yolo5_fashion2/results.csv
```

---

## 🔄 이어서 학습 (맥북에서)

전송 후 맥북에서 이어서 학습:

```bash
cd /Users/jimin/opensw/FItzy_copy
python train_fashion.py \
  --resume \
  --resume-from runs/train/yolo5_fashion2/weights/last.pt \
  --epochs 100 \
  --batch 32 \
  --device 0
```

---

## ⚠️ 주의사항

1. **파일 경로**: 맥북의 프로젝트 경로는 `/Users/jimin/opensw/FItzy_copy/`로 확인됨
2. **권한**: 파일 복사 시 권한 문제가 있을 수 있으니 확인 필요
3. **덮어쓰기**: 기존 파일이 있으면 확인 후 덮어쓰기

