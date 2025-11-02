# 추천 코디 AI 이미지 생성 도구 가이드

## 개요
각 추천 코디에 대한 AI 생성 이미지를 만드는데 사용할 수 있는 도구들을 정리하고 통합 방법을 제시합니다.

---

## 추천 도구 비교

### 1. Stable Diffusion (Hugging Face) ⭐ **추천**

**장점:**
- ✅ **무료 사용 가능** (로컬 실행)
- ✅ **오픈소스** - 커스터마이징 자유
- ✅ **Python 통합 용이** - `diffusers` 라이브러리
- ✅ **패션 특화 모델** 존재 (FashioniGen 등)
- ✅ **ControlNet** 사용 가능 - 의상 구조 제어

**단점:**
- ⚠️ GPU 메모리 필요 (최소 4GB VRAM)
- ⚠️ 로컬 실행 시 리소스 사용 큼
- ⚠️ API 버전은 유료

**통합 난이도:** ⭐⭐ (중)

**비용:**
- 로컬 실행: 무료
- Hugging Face Inference API: 무료 티어 있음 (제한적)
- Replicate API: $0.0023/이미지

**코드 예시:**
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "fashion outfit: red shirt, blue jeans, casual style"
image = pipe(prompt).images[0]
```

---

### 2. DALL-E (OpenAI) ⭐ **간단함**

**장점:**
- ✅ **API 제공** - 구현 간단
- ✅ **고품질 이미지**
- ✅ **안정적 서비스**
- ✅ **다양한 스타일 지원**

**단점:**
- ❌ **유료** ($0.040 ~ $0.120 per image)
- ❌ **커스터마이징 제한적**
- ❌ **패션 특화 기능 없음**

**통합 난이도:** ⭐ (쉬움)

**비용:**
- DALL-E 2: $0.020 per image (1024x1024)
- DALL-E 3: $0.040 per image (1024x1024 standard), $0.120 (HD)

**코드 예시:**
```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.images.generate(
    model="dall-e-3",
    prompt="fashion outfit: red shirt, blue jeans, casual style, full body",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
```

---

### 3. Stability AI API (Stable Diffusion API)

**장점:**
- ✅ **API 제공** - 구현 간단
- ✅ **Stable Diffusion 기반**
- ✅ **무료 티어** (월 10 credits)
- ✅ **다양한 모델 선택 가능**

**단점:**
- ⚠️ **유료** (무료 티어 제한적)
- ⚠️ **API 비용** ($0.01 ~ $0.04 per image)

**통합 난이도:** ⭐ (쉬움)

**비용:**
- 무료 티어: 월 10 credits (약 10장)
- 유료: $0.01 per image (512x512), $0.04 (1024x1024)

**코드 예시:**
```python
import requests

api_key = "your-api-key"
response = requests.post(
    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "text_prompts": [{"text": "fashion outfit: red shirt, blue jeans"}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
    }
)

image_data = response.json()["artifacts"][0]["base64"]
```

---

### 4. ControlNet for Fashion (고급)

**장점:**
- ✅ **의상 구조 제어 가능**
- ✅ **정확한 포즈/실루엣 제어**
- ✅ **패션 특화 기능**

**단점:**
- ⚠️ **구현 복잡**
- ⚠️ **GPU 필요**
- ⚠️ **로컬 실행 필요**

**통합 난이도:** ⭐⭐⭐ (어려움)

**비용:** 무료 (로컬 실행)

---

### 5. Hugging Face Inference API (간단한 통합)

**장점:**
- ✅ **간단한 API 호출**
- ✅ **다양한 모델 선택**
- ✅ **무료 티어** (제한적)

**단점:**
- ⚠️ **무료 티어 제한** (분당 요청 수 제한)
- ⚠️ **대기 시간 발생 가능**

**통합 난이도:** ⭐ (쉬움)

**코드 예시:**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="your-token")

image = client.text_to_image(
    "fashion outfit: red shirt, blue jeans, casual style",
    model="runwayml/stable-diffusion-v1-5"
)
```

---

## 패션 특화 모델 추천

### 1. **FashioniGen** (Stable Diffusion 기반)
- 패션 아이템 생성에 특화
- 의상, 액세서리, 신발 등 세밀한 제어 가능

### 2. **Fashion-ICON** 
- 의상 아이콘 생성
- 간단한 스케치 스타일

### 3. **ControlNet + OpenPose**
- 인체 포즈 제어
- 정확한 의상 착용 이미지 생성

---

## 추천 순위

### 🥇 1순위: **Stable Diffusion (Hugging Face diffusers)** + **Hugging Face Inference API**
- **이유**: 
  - 로컬 실행 가능 (무료)
  - API로도 사용 가능
  - 패션 특화 모델 활용 가능
  - 오픈소스 커뮤니티 활발

### 🥈 2순위: **DALL-E API (OpenAI)**
- **이유**:
  - 구현이 가장 간단
  - 안정적 서비스
  - 고품질 결과
  - 빠른 응답

### 🥉 3순위: **Stability AI API**
- **이유**:
  - Stable Diffusion 기반
  - 무료 티어 제공
  - API 사용 간편

---

## 통합 전략

### 옵션 1: 하이브리드 접근 (추천)

```python
# 1순위: 로컬 Stable Diffusion (무료)
# 2순위: Hugging Face Inference API (무료 티어)
# 3순위: DALL-E API (유료, fallback)
```

### 옵션 2: 단순 통합 (빠른 구현)

```python
# DALL-E API만 사용
# 간단하지만 비용 발생
```

### 옵션 3: 로컬 전용 (비용 절감)

```python
# Stable Diffusion 로컬 실행
# GPU 필요, 하지만 완전 무료
```

---

## 구현 예시 코드 구조

```python
# src/utils/image_generator.py

class OutfitImageGenerator:
    """추천 코디 AI 이미지 생성 클래스"""
    
    def __init__(self, method="stable_diffusion"):
        self.method = method
        # 초기화 로직
    
    def generate_outfit_image(self, outfit_description, style_info):
        """코디 설명을 바탕으로 이미지 생성"""
        prompt = self._build_prompt(outfit_description, style_info)
        
        if self.method == "dall_e":
            return self._generate_with_dalle(prompt)
        elif self.method == "stable_diffusion":
            return self._generate_with_sd(prompt)
        elif self.method == "huggingface_api":
            return self._generate_with_hf_api(prompt)
    
    def _build_prompt(self, outfit_description, style_info):
        """효과적인 프롬프트 생성"""
        # 의상 아이템 + 색상 + 스타일 + 품질 키워드
        pass
```

---

## 프롬프트 작성 가이드

### 좋은 프롬프트 예시:
```
"Professional fashion photography, full body shot, 
{color} {item_type}, {style} style, 
high quality, detailed, fashion magazine style,
neutral background, studio lighting"
```

### 나쁜 프롬프트 예시:
```
"red shirt"  # 너무 단순
```

---

## 비용 비교표

| 도구 | 무료 티어 | 유료 비용 | 품질 | 속도 |
|------|----------|----------|------|------|
| Stable Diffusion (로컬) | ✅ 무제한 | 무료 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Hugging Face API | ✅ 제한적 | 무료~유료 | ⭐⭐⭐ | ⭐⭐ |
| DALL-E 3 | ❌ | $0.040/image | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Stability AI API | ✅ 10/month | $0.01/image | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 결론 및 추천

**현재 프로젝트에 가장 적합한 선택:**

1. **개발/테스트 단계**: Hugging Face Inference API (무료 티어)
2. **프로덕션**: 
   - GPU 있는 서버: Stable Diffusion 로컬
   - GPU 없는 서버: DALL-E API 또는 Stability AI API

**추천 구현 순서:**
1. Hugging Face Inference API로 프로토타입 구현
2. 비용/품질 비교 후 최종 결정
3. Stable Diffusion 로컬로 전환 (GPU 가능 시)

