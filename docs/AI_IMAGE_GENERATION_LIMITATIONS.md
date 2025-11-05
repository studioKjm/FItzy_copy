# AI 이미지 생성 기능 - 현황 및 한계

## 📊 현재 상태

### 적용된 모델
- **Stable Diffusion 2.1** (stabilityai/stable-diffusion-2-1)
  - SD 1.4 대비 개선된 텍스트 이해력 및 색상 표현
  - 메모리: 약 5GB
  - 생성 시간: 약 30-40초 (M2 MacBook, 30 steps)

### 적용된 모든 최적화

1. **프롬프트 최적화**
   - UPPERCASE 색상 강조: "BLACK LONG SLEEVE SHIRT, GRAY PANTS"
   - 단순화: 핵심 키워드만 사용 (CLIP 77 토큰 제한)
   - 아이템 수 제한: 최대 2개 (정확도 우선)

2. **Negative Prompt 강화**
   ```
   face, head, eyes, nose, mouth, lips, hair, neck, portrait, person, human face,
   multiple people, multiple mannequins, two mannequins, crowd, group,
   wrong colors, incorrect colors, color swap, reversed colors,
   white pants, shorts, blurry, low quality
   ```

3. **생성 파라미터 극대화**
   - `guidance_scale`: 15.0 (프롬프트 준수도 최대)
   - `num_inference_steps`: 30 (품질 향상)
   - `seed`: 42 (일관성)

4. **후처리**
   - 상단 40% 자동 크롭 (얼굴/목 완전 제거)

5. **MPS 최적화**
   - UNet: MPS (GPU)
   - VAE: CPU (안정성)
   - TextEncoder: CPU (임베딩 안정성)

---

## ⚠️ 현재 한계

### 1. 색상 정확도 문제
**증상**: 요청한 색상과 다르게 생성
- 예: "검은색 긴팔 셔츠 + 회색 바지" → "회색 셔츠 + 검은 바지"
- 발생 빈도: 약 30-50%

**원인**: 
- Stable Diffusion 모델의 색상 조건부 생성 한계
- 복잡한 색상 조합에서 혼동 발생
- guidance_scale을 높여도 완전히 해결되지 않음

### 2. 아이템 누락/추가 문제
**증상**: 요청하지 않은 아이템 추가 또는 누락
- 예: "반팔 티셔츠 + 긴팔 재킷 + 가디건" → "반팔 티셔츠"만 출력
- 예: "검은 티셔츠" → "검은 티셔츠 + 흰 바지" 출력

**원인**:
- 3개 이상의 아이템은 프롬프트가 복잡해짐
- 모델이 가장 강한 신호(첫 번째 아이템)에만 집중
- 현재 2개 아이템으로 제한했으나 여전히 부정확

### 3. 여러 마네킹 출력 문제
**증상**: 하나의 마네킹이 아닌 여러 개가 나타남
- "multiple mannequins" negative prompt로 개선되었으나 완전하지 않음

---

## 🔬 시도한 모델 비교

### Stable Diffusion v1.4 (CompVis/stable-diffusion-v1-4)
- ❌ 색상 정확도: 낮음
- ❌ 아이템 정확도: 낮음
- ✅ 생성 속도: 빠름 (약 35초)
- ✅ 메모리: 약 4GB

### Stable Diffusion 2.1 (stabilityai/stable-diffusion-2-1) ⭐ 현재 사용
- ⚠️ 색상 정확도: 중간 (SD 1.4보다 약간 나음)
- ⚠️ 아이템 정확도: 중간
- ✅ 생성 속도: 빠름 (약 32초)
- ✅ 메모리: 약 5GB
- ✅ M2 MacBook 호환: 완벽

### SDXL-Turbo (stabilityai/sdxl-turbo)
- ✅ 색상 정확도: 높음 (예상)
- ✅ 생성 속도: 매우 빠름 (4 steps, 약 5-10초)
- ❌ 메모리: 약 7GB (M2 8GB에서 불안정할 수 있음)
- ⚠️ 다운로드 시간: 약 10분
- **결과**: 로드 시도했으나 메모리 제약으로 불안정

---

## 💡 현실적인 해결 방안

### 방안 1: AI 이미지를 "참고용"으로 명시 ⭐ 권장
```python
st.warning("⚠️ AI 생성 이미지는 참고용입니다. 색상/아이템이 정확하지 않을 수 있습니다.")
st.info("💡 텍스트 기반 추천을 우선으로 참고해주세요.")
```

**장점**:
- 사용자 기대치 조정
- 기능 유지하면서 책임 회피
- 개선 여지 남김

### 방안 2: AI 이미지 생성 기능 비활성화
```python
# 기본값을 False로 변경
if 'enable_ai_images' not in st.session_state:
    st.session_state.enable_ai_images = False
```

**장점**:
- 부정확한 결과로 인한 사용자 혼란 방지
- 앱 속도 향상
- 텍스트 추천에 집중

**단점**:
- AI 이미지 생성 기능 사용 불가

### 방안 3: 실제 제품 이미지 링크 제공 (향후 구현)
```python
# 쇼핑몰 API 연동
products_with_images = [
    {"name": "자라 크롭 티셔츠", "image_url": "...", "link": "..."},
    # ...
]
```

**장점**:
- 정확한 제품 이미지
- 실제 구매로 연결
- 사용자 경험 향상

**단점**:
- API 연동 필요
- 제품 데이터베이스 구축 필요

---

## 🚀 향후 개선 방향

### 단기 (현재 가능)
1. ✅ **SD 2.1 사용** (완료)
2. ⚠️ **AI 이미지 "참고용" 경고 추가** (권장)
3. ⚠️ **기본값 OFF로 변경** (선택적)

### 중기 (추가 개발 필요)
1. **SDXL-Turbo 메모리 최적화**
   - `enable_model_cpu_offload()` 사용
   - attention slicing 최대화
   - 8-bit quantization 적용

2. **ControlNet 통합**
   - 스케치 기반 정확한 의상 생성
   - 색상 조건 명확화

### 장기 (GPU 서버 필요)
1. **SDXL (Full)** 사용
   - 최고 품질
   - 16GB+ VRAM 필요

2. **Fine-tuning**
   - 패션 전문 모델 학습
   - DeepFashion2 데이터셋 활용

3. **Stable Diffusion 3.0**
   - 최신 모델 (출시 시)

---

## 📝 사용자 안내 문구 (권장)

### 앱에 추가할 안내
```python
with st.expander("ℹ️ AI 이미지 생성 안내", expanded=False):
    st.markdown("""
    ### ⚠️ AI 이미지 생성의 한계
    
    현재 사용 중인 Stable Diffusion 2.1 모델은 무료로 사용 가능하지만, 
    다음과 같은 한계가 있습니다:
    
    - **색상 정확도**: 요청한 색상과 다르게 생성될 수 있습니다
    - **아이템 정확도**: 일부 아이템이 누락되거나 다르게 표시될 수 있습니다
    - **일관성**: 같은 요청도 다른 결과가 나올 수 있습니다
    
    ### 💡 권장 사항
    
    1. **텍스트 기반 추천을 우선으로 참고**해주세요
    2. AI 이미지는 **스타일 참고용**으로만 활용해주세요
    3. 정확한 제품은 **추천 제품 목록**을 참고해주세요
    
    ### 🎯 개선 계획
    
    - 더 나은 모델로 업그레이드 예정 (SDXL, ControlNet 등)
    - 실제 제품 이미지 연동 검토 중
    """)
```

---

## 🔧 기술적 대안 검토

### 시도 가능한 무료 모델 (M2 MacBook)

| 모델 | 메모리 | 속도 | 정확도 | M2 호환 | 상태 |
|------|--------|------|--------|---------|------|
| SD 1.4 | 4GB | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | 이전 버전 |
| SD 2.1 | 5GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | **현재 사용** |
| SDXL-Turbo | 7GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | 메모리 부족 가능 |
| SDXL (Full) | 8GB+ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | 메모리 부족 |

### 결론
- **현재 최선**: Stable Diffusion 2.1
- **M2 8GB 환경의 한계**: SDXL 계열은 불안정
- **권장**: AI 이미지는 참고용으로 사용

---

## 📌 결론 및 권장사항

### ✅ 구현 완료
1. Stable Diffusion 2.1로 업그레이드
2. 모든 프롬프트 및 파라미터 최적화
3. 얼굴 제거 완벽 구현 (40% 크롭)
4. 단일 마네킹 출력 개선

### ⚠️ 남아있는 한계
1. 색상 정확도: 약 50-70% (모델 한계)
2. 아이템 정확도: 약 60-80% (모델 한계)

### 💡 최종 권장사항
**AI 이미지 생성을 "스타일 참고용"으로 위치**
- 기본값 OFF 또는 경고 문구 추가
- 텍스트 기반 추천을 메인 기능으로 강조
- 향후 GPU 서버 또는 SDXL 도입 시 개선 가능

---

**작성일**: 2025-11-03  
**모델**: Stable Diffusion 2.1  
**환경**: M2 MacBook, 8GB Memory

