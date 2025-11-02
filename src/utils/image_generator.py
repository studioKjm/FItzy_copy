"""
추천 코디 AI 이미지 생성 유틸리티
Stable Diffusion, DALL-E 등 다양한 이미지 생성 모델 지원
"""

import os
from PIL import Image
import io
import requests
from typing import Optional, Dict, List


class OutfitImageGenerator:
    """추천 코디 AI 이미지 생성 클래스"""
    
    def __init__(self, method: str = "huggingface_api", use_prototype: bool = True):
        """
        Args:
            method: 이미지 생성 방법
                - "huggingface_api": Hugging Face Inference API (추천, 무료 티어)
                - "dall_e": OpenAI DALL-E API (유료, 고품질)
                - "stable_diffusion": 로컬 Stable Diffusion (무료, GPU 필요)
                - "stability_ai": Stability AI API (유료)
        """
        self.method = method
        self.api_key = None
        self.use_prototype = use_prototype  # 프로토타입 사용 여부
        
        # API 키 설정
        if method == "dall_e":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif method == "stability_ai":
            self.api_key = os.getenv("STABILITY_AI_API_KEY")
        elif method == "huggingface_api":
            # Hugging Face API 키는 선택적 (무료 티어는 키 없이도 사용 가능)
            # 빈 문자열이나 공백만 있으면 None으로 처리
            api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
            self.api_key = api_key if api_key else None
        
        # 프로토타입 매니저 초기화 (Stable Diffusion 로컬만 지원)
        if use_prototype and method == "stable_diffusion":
            try:
                from src.utils.face_prototype_manager import FacePrototypeManager
                self.prototype_manager = FacePrototypeManager()
            except:
                self.use_prototype = False
                self.prototype_manager = None
        else:
            self.prototype_manager = None
    
    def generate_outfit_image(self, outfit_description: Dict, style_info: Dict = None) -> Optional[Image.Image]:
        """
        코디 설명을 바탕으로 AI 이미지 생성
        프로토타입이 있으면 의상만 변경하여 빠르게 생성
        
        Args:
            outfit_description: 코디 설명 딕셔너리
                - items: 아이템 리스트 (예: ["빨간 상의", "파란 바지"])
                - style: 스타일 (예: "캐주얼")
                - colors: 색상 리스트
                - gender: 성별 ("남성", "여성", "공용")
            style_info: 추가 스타일 정보
        
        Returns:
            생성된 이미지 (PIL Image) 또는 None
        """
        # 프로토타입 사용 가능 여부 확인
        gender = outfit_description.get("gender", "공용")
        use_prototype_mode = (
            self.use_prototype and 
            self.method == "stable_diffusion" and 
            self.prototype_manager and
            gender != "공용"  # 공용은 프로토타입 미사용
        )
        
        if use_prototype_mode:
            # 프로토타입 기반 이미지 생성 (의상만 변경)
            return self._generate_with_prototype(outfit_description, style_info)
        else:
            # 기존 방식 (전체 이미지 생성)
            prompt = self._build_prompt(outfit_description, style_info)
            try:
                if self.method == "huggingface_api":
                    return self._generate_with_huggingface_api(prompt)
                elif self.method == "dall_e":
                    return self._generate_with_dalle(prompt)
                elif self.method == "stable_diffusion":
                    return self._generate_with_stable_diffusion_local(prompt)
                elif self.method == "stability_ai":
                    return self._generate_with_stability_ai(prompt)
                else:
                    print(f"⚠️ 알 수 없는 방법: {self.method}")
                    return None
            except Exception as e:
                print(f"⚠️ 이미지 생성 실패: {e}")
                return None
    
    def _build_prompt(self, outfit_description: Dict, style_info: Dict = None) -> str:
        """효과적인 프롬프트 생성 (CLIP 토크나이저 77 토큰 제한 고려)"""
        items = outfit_description.get("items", [])
        style = outfit_description.get("style", "캐주얼")
        colors = outfit_description.get("colors", [])
        gender = outfit_description.get("gender", "공용")  # 성별 정보 가져오기
        
        # 성별 키워드 결정
        if gender == "남성":
            gender_keyword = "male model, man"
        elif gender == "여성":
            gender_keyword = "female model, woman"
        else:
            gender_keyword = "model"  # 공용 또는 미지정
        
        # 아이템과 색상을 정확하게 매핑
        # items가 "검은색 긴팔 상의" 형식이면 그대로 사용
        # items가 일반 이름이면 colors와 결합
        
        # 한국어 색상 → 영어 변환
        color_map = {
            "검은색": "black", "흰색": "white", "빨간색": "red", "파란색": "blue",
            "노란색": "yellow", "초록색": "green", "보라색": "purple", "분홍색": "pink",
            "회색": "gray", "갈색": "brown", "베이지": "beige", "파스텔": "pastel",
            "black": "black", "white": "white", "red": "red", "blue": "blue"
        }
        
        # 아이템 처리: 이미 색상이 포함된 경우와 그렇지 않은 경우
        # 최대 3개까지 포함 (재킷, 가디건 등 레이어드 아이템 표현)
        processed_items = []
        for item in items[:3]:  # 최대 3개로 증가
            item_lower = item.lower()
            
            # 제품명/브랜드명 제거 (의상 설명만 사용)
            # "유니클로", "리바이스", "컨버스" 등 브랜드명 제거
            brand_keywords = ["유니클로", "리바이스", "컨버스", "나이키", "아디다스", "uniqlo", "levis", "converse", "nike", "adidas", "u ", "U "]
            for brand in brand_keywords:
                item_lower = item_lower.replace(brand, "")
                item = item.replace(brand, "").replace(brand.capitalize(), "").replace(brand.upper(), "")
            
            # 불필요한 제품명 키워드 제거
            product_keywords = ["크루넥", "crew neck", "u 크루넥", "511", "척테일러", "chuck taylor", "슬림진", "slim", "스탠스미스", "stansmith", "아크테릭스", "arcteryx", "테크플리스", "tech fleece", "살로몬", "salomon", "xt-6"]
            for keyword in product_keywords:
                if keyword in item_lower:
                    # 제품명은 제거하고 의상 타입만 남김
                    item = item.replace(keyword, "").strip()
            
            # 바지 타입 명확화 (반바지 방지)
            if "바지" in item_lower or "pants" in item_lower:
                if "반바지" not in item_lower and "shorts" not in item_lower:
                    # 긴바지로 명시
                    item = item.replace("바지", "long pants").replace("pants", "long pants")
            
            # 액세서리 타입 명확화
            if "액세서리" in item_lower or "accessory" in item_lower:
                # 일반적인 액세서리는 모자/캡으로 구체화 (기본값)
                item = item.replace("액세서리", "cap").replace("accessory", "cap")
            
            # 정리: 공백 제거 및 의상 타입 명확화
            item = " ".join(item.split())  # 중복 공백 제거
            
            # "앞치마", "에이프런" 같은 키워드가 혼동되는 것을 방지
            if "앞치마" in item_lower or "에이프런" in item_lower or "apron" in item_lower:
                # 앞치마 관련 키워드 제거
                item = item.replace("앞치마", "").replace("에이프런", "").replace("apron", "").strip()
            
            # 색상이 포함되어 있는지 확인
            has_color = any(color in item_lower for color in ["검은색", "흰색", "빨간색", "파란색", "black", "white", "red", "blue", "파스텔", "pastel"])
            if not has_color and colors:
                # 색상이 없으면 첫 번째 색상 추가
                color_en = color_map.get(colors[0], colors[0])
                processed_items.append(f"{color_en} {item}")
            else:
                # 이미 색상이 있으면 영어로 변환만
                for kr_color, en_color in color_map.items():
                    item = item.replace(kr_color, en_color)
                # 공백 정리
                item = " ".join(item.split())
                if item:  # 빈 문자열이 아닌 경우만 추가
                    processed_items.append(item)
        
        # 아이템을 레이어드 방식으로 표현 (재킷, 가디건이 상의 위에 입혀진 형태)
        # 추천 상품 정보를 고려하여 더 정확한 표현
        if len(processed_items) > 1:
            # 첫 번째 아이템이 상의, 나머지가 외투/재킷/가디건인 경우
            main_item = processed_items[0]
            outer_items = ", ".join(processed_items[1:])
            items_text = f"{main_item} with {outer_items} over it"
        else:
            items_text = ", ".join(processed_items) if processed_items else "fashion outfit"
        
        # 추천 상품과 매칭을 위한 추가 정보 확인
        # style_info에서 추천 상품 정보가 있다면 더 구체적으로 표현
        if style_info:
            # 추천 상품 타입을 고려하여 의상 타입 명확화
            # 예: "티셔츠" → "t-shirt", "재킷" → "jacket" 등
            items_text = items_text.replace("상의", "top").replace("하의", "pants").replace("티셔츠", "t-shirt").replace("재킷", "jacket")
        
        # 스타일 키워드 (간결하게)
        style_keywords = {
            "캐주얼": "casual style",
            "포멀": "formal elegant style",
            "트렌디": "trendy modern style",
            "스포츠": "sporty athletic style",
            "빈티지": "vintage retro style",
            "모던": "modern contemporary style"
        }
        style_en = style_keywords.get(style, "casual style")
        
        # 간결한 프롬프트 구성 (77 토큰 제한 고려)
        # 목 아래만 출력 (얼굴 제거), 의상 명확하게 표현
        # 제품명은 포함하지 않고 의상 설명만 사용
        
        # 바지 길이 명시 (반바지 방지)
        items_text_processed = items_text
        if "바지" in items_text.lower() or "pants" in items_text.lower():
            if "반바지" not in items_text.lower() and "shorts" not in items_text.lower():
                items_text_processed = items_text.replace("바지", "long pants").replace("pants", "long pants")
        
        # 액세서리 명시 (모자, 캡 등)
        accessory_keywords = ["액세서리", "accessory", "캡", "cap", "모자", "hat"]
        has_accessory = any(kw in items_text.lower() for kw in accessory_keywords)
        if has_accessory:
            items_text_processed += ", wearing cap"
        
        # 목 아래만 출력 (얼굴 제거), 의상 중심
        # 얼굴 관련 키워드 모두 제거하고 목 아래부터 강조
        prompt = f"Fashion photography, {gender_keyword} wearing {items_text_processed}, {style_en}, neck down only, upper body and full body visible, entire outfit visible, legs visible, standing pose, no face visible, head cropped out, focus on clothing, high quality, fashion magazine style, neutral background, studio lighting, 8k"
        
        return prompt
    
    def _generate_with_huggingface_api(self, prompt: str) -> Optional[Image.Image]:
        """Hugging Face Inference API 사용 (추천 - 무료 티어)"""
        import time
        
        # 모델 목록 (우선순위 순, 실제로 작동하는 모델 우선)
        # 최근 Hugging Face 정책 변경으로 일부 모델 접근 불가
        # 작동하는 모델로 업데이트
        models_to_try = [
            "stabilityai/stable-diffusion-2-1",  # 안정적인 모델
            "runwayml/stable-diffusion-v1-5",   # 원래 모델
            "CompVis/stable-diffusion-v1-4"     # 대안
        ]
        
        # 재시도 설정 (최대 3회, 503 에러 시 모델 로딩 대기)
        max_retries = 3
        retry_delay = 5  # 초
        current_model_idx = 0
        
        # 첫 번째 모델로 시작
        model = models_to_try[current_model_idx]
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json"
                }
                # API 키가 제공된 경우에만 사용 (무료 티어는 키 없이도 사용 가능)
                if self.api_key and self.api_key.strip():
                    headers["Authorization"] = f"Bearer {self.api_key.strip()}"
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "num_inference_steps": 25,  # 속도와 품질 균형
                        "guidance_scale": 7.5
                    }
                }
                
                # 요청 전송 (타임아웃 설정)
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    json=payload,
                    timeout=90  # 모델 로딩을 고려하여 90초로 증가
                )
                
                # 응답 처리
                if response.status_code == 200:
                    image_bytes = response.content
                    if image_bytes:
                        image = Image.open(io.BytesIO(image_bytes))
                        return image
                    else:
                        print("⚠️ 이미지 데이터가 비어있습니다.")
                        return None
                        
                elif response.status_code == 503:
                    # 모델이 로딩 중인 경우 - 재시도
                    error_info = {}
                    try:
                        error_info = response.json()
                    except:
                        pass
                    
                    estimated_time = error_info.get("estimated_time", retry_delay)
                    if attempt < max_retries - 1:
                        wait_time = min(int(estimated_time) if isinstance(estimated_time, (int, float)) else retry_delay, 30)
                        print(f"⏳ 모델 로딩 중... (예상 대기: {wait_time}초, 시도 {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"⚠️ 모델 로딩 시간이 초과되었습니다. ({estimated_time}초 예상)")
                        print("💡 잠시 후 수동으로 다시 시도해주세요.")
                        return None
                        
                elif response.status_code == 401:
                    # 인증 오류 - 다른 모델로 시도하거나 재시도
                    error_info = {}
                    try:
                        error_info = response.json()
                    except:
                        pass
                    
                    # 다른 모델로 시도
                    if current_model_idx < len(models_to_try) - 1 and attempt < max_retries - 1:
                        current_model_idx += 1
                        model = models_to_try[current_model_idx]
                        api_url = f"https://api-inference.huggingface.co/models/{model}"
                        print(f"⚠️ 모델 인증 오류. 다른 모델로 시도합니다: {model}")
                        time.sleep(2)
                        continue
                    # 잘못된 키가 있으면 제거하고 재시도 (한 번만)
                    elif self.api_key and attempt == 0:
                        print("⚠️ Hugging Face API 키가 잘못되었습니다. 키 없이 재시도합니다...")
                        self.api_key = None  # 키 제거
                        current_model_idx = 0  # 첫 번째 모델로 리셋
                        api_url = f"https://api-inference.huggingface.co/models/{models_to_try[0]}"
                        time.sleep(2)  # 짧은 대기 후 재시도
                        continue
                    else:
                        print("⚠️ Hugging Face API 인증 오류 (401)")
                        if not self.api_key:
                            print("💡 무료 티어 사용 중입니다.")
                            print("💡 최근 정책 변경으로 모든 모델에 API 키가 필요합니다.")
                        else:
                            print("💡 API 키 권한을 확인해주세요.")
                        print("💡 해결 방법:")
                        print("   1. Hugging Face 계정 생성 (무료)")
                        print("   2. API 토큰 생성 (Read 권한)")
                        print("      🔗 https://huggingface.co/settings/tokens")
                        print("   3. 앱 사이드바에서 API 키 입력")
                        return None
                        
                elif response.status_code == 403:
                    # 권한 부족 오류
                    error_info = {}
                    try:
                        error_info = response.json()
                    except:
                        pass
                    
                    error_message = error_info.get("error", "권한 부족")
                    print(f"⚠️ Hugging Face API 권한 오류 (403): {error_message}")
                    print("💡 현재 API 토큰에 Inference API 사용 권한이 없습니다.")
                    print()
                    print("💡 해결 방법:")
                    print("   1. Hugging Face 사이트에서 기존 토큰 삭제")
                    print("   2. 새 토큰 생성 시 'Read' 권한 선택 (필수)")
                    print("   3. 또는 프로 유저로 업그레이드 (유료)")
                    print("      🔗 https://huggingface.co/settings/tokens")
                    print()
                    print("💡 대안:")
                    print("   - 다른 이미지 생성 방법 사용 (DALL-E, Stable Diffusion 로컬 등)")
                    print("   - 이미지 생성 기능 비활성화 후 텍스트 기반 추천만 사용")
                    return None
                        
                elif response.status_code == 429:
                    # 요청 한도 초과
                    retry_after = response.headers.get("Retry-After", retry_delay)
                    if attempt < max_retries - 1:
                        wait_time = min(int(retry_after) if retry_after.isdigit() else retry_delay, 30)
                        print(f"⏳ 요청 한도 초과. {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("⚠️ API 요청 한도 초과")
                        print("💡 무료 티어는 분당 요청 수 제한이 있습니다. 잠시 후 다시 시도해주세요.")
                        return None
                        
                elif response.status_code == 404:
                    # 모델을 찾을 수 없음 - 다른 모델로 시도
                    error_info = {}
                    try:
                        error_info = response.json()
                    except:
                        pass
                    
                    error_message = error_info.get("error", "모델을 찾을 수 없습니다")
                    print(f"⚠️ Hugging Face API 오류 (404): {error_message}")
                    
                    # 다른 모델로 시도
                    if current_model_idx < len(models_to_try) - 1 and attempt < max_retries - 1:
                        current_model_idx += 1
                        model = models_to_try[current_model_idx]
                        api_url = f"https://api-inference.huggingface.co/models/{model}"
                        print(f"⚠️ 모델을 찾을 수 없습니다. 다른 모델로 시도합니다: {model}")
                        time.sleep(2)
                        continue
                    else:
                        print("⚠️ 사용 가능한 모델을 찾을 수 없습니다.")
                        print("💡 Hugging Face의 정책이 변경되었거나 모델이 비공개로 전환되었습니다.")
                        print("💡 해결 방법:")
                        print("   1. **DALL-E API 사용** (가장 안정적, 유료)")
                        print("   2. **Stable Diffusion 로컬 실행** (무료, GPU 필요)")
                        print("   3. **이미지 생성 기능 비활성화** (텍스트 추천만 사용)")
                        print("   4. **Hugging Face Pro 계정 업그레이드** (유료)")
                        return None
                        
                else:
                    # 기타 오류
                    error_info = {}
                    try:
                        if response.headers.get("content-type", "").startswith("application/json"):
                            error_info = response.json()
                    except:
                        pass
                    
                    error_message = error_info.get("error", response.text[:200] if response.text else "알 수 없는 오류")
                    print(f"⚠️ Hugging Face API 오류 ({response.status_code}): {error_message}")
                    
                    # 일부 오류는 재시도 가능
                    if response.status_code >= 500 and attempt < max_retries - 1:
                        print(f"⏳ 서버 오류. {retry_delay}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    
                    return None
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏳ 요청 시간 초과. {retry_delay}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("⚠️ 요청 시간 초과 (최대 재시도 횟수 초과)")
                    print("💡 네트워크를 확인하거나 잠시 후 다시 시도해주세요.")
                    return None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ 오류 발생: {str(e)[:100]}. {retry_delay}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"⚠️ Hugging Face API 호출 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
        
        return None
    
    def _generate_with_dalle(self, prompt: str) -> Optional[Image.Image]:
        """OpenAI DALL-E API 사용 (유료, 고품질)"""
        try:
            from openai import OpenAI
            
            if not self.api_key:
                print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
                return None
            
            client = OpenAI(api_key=self.api_key)
            
            # DALL-E 3 사용 (더 나은 품질)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # URL에서 이미지 다운로드
            img_response = requests.get(image_url)
            image = Image.open(io.BytesIO(img_response.content))
            
            return image
        except ImportError:
            print("⚠️ openai 라이브러리가 설치되지 않았습니다: pip install openai")
            return None
        except Exception as e:
            print(f"⚠️ DALL-E API 호출 실패: {e}")
            return None
    
    def _generate_with_prototype(self, outfit_description: Dict, style_info: Dict = None) -> Optional[Image.Image]:
        """프로토타입 기반 이미지 생성 (의상만 변경, 빠른 속도)"""
        gender = outfit_description.get("gender", "공용")
        
        # 프로토타입 로드
        prototype = self.prototype_manager.load_prototype(gender)
        
        # 프로토타입이 없으면 생성
        if not prototype:
            print(f"💡 {gender} 프로토타입이 없습니다. 생성 중... (한 번만 생성됨)")
            prototype = self.prototype_manager.generate_prototype(gender, self)
            if not prototype:
                print("⚠️ 프로토타입 생성 실패. 전체 이미지 생성으로 전환합니다.")
                prompt = self._build_prompt(outfit_description, style_info)
                return self._generate_with_stable_diffusion_local(prompt)
        
        # 의상 변경 프롬프트 생성
        prompt = self._build_prompt(outfit_description, style_info)
        
        # img2img 방식으로 의상만 변경
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            import torch
            import numpy as np
            
            # 장치 설정
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                vae_device = "cpu"
                dtype = torch.float16
            elif torch.cuda.is_available():
                device = "cuda"
                vae_device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                vae_device = "cpu"
                dtype = torch.float32
            
            model_name = "CompVis/stable-diffusion-v1-4"
            
            print("🎨 프로토타입 기반 의상 변경 중... (빠른 속도)")
            print(f"⏳ 예상 시간: 약 15-30초 (최적화됨)")
            
            # Img2Img 파이프라인 로드
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if device == "mps":
                pipe.vae.to(vae_device)
                pipe.unet.to(device)
                pipe.text_encoder.to(device)
                pipe.enable_attention_slicing()
            else:
                pipe = pipe.to(device)
            
            # 프로토타입을 초기 이미지로 사용
            # 얼굴 완전 제거, 목 아래만, 하반신 포함
            # 얼굴 관련 모든 키워드 포함 (완전히 방지)
            negative_prompt = "face, head, facial features, eyes, nose, mouth, chin, forehead, cheek, ear, hair, face visible, showing face, portrait, headshot, close-up face, cropped legs, missing legs, cut off at waist, upper body only, shorts, short pants, cropped pants, blurry, watermark, grainy, signature, cut off, draft, low quality, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, mutation, deformed, bad body, blurry, bad anatomy, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, jpeg artifacts, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
            
            # img2img 생성 (의상만 변경, 얼굴은 유지)
            # strength 조정: 너무 높으면 얼굴이 망가질 수 있음
            # 의상 변경과 얼굴 보존의 균형
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    image=prototype,
                    negative_prompt=negative_prompt,
                    strength=0.7,  # 0.7로 증가 (의상 변경 확실히, 얼굴은 어차피 안 보임)
                    num_inference_steps=15,  # 속도 최적화 (15단계로 감소)
                    guidance_scale=7.0,  # 빠른 생성
                )
            
            image = result.images[0]
            print("✅ 프로토타입 기반 이미지 생성 완료!")
            return image
            
        except ImportError:
            print("⚠️ diffusers 라이브러리가 설치되지 않았습니다.")
            print("💡 설치 방법: pip install diffusers accelerate")
            return None
        except Exception as e:
            print(f"⚠️ 프로토타입 기반 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            # fallback: 전체 이미지 생성
            print("💡 전체 이미지 생성으로 전환합니다.")
            prompt = self._build_prompt(outfit_description, style_info)
            return self._generate_with_stable_diffusion_local(prompt)
    
    def _generate_with_stable_diffusion_local(self, prompt: str, negative_prompt: str = None) -> Optional[Image.Image]:
        """로컬 Stable Diffusion 사용 (무료, M2 맥북 최적화)"""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            # Apple Silicon (M1/M2) 최적화
            # MPS는 VAE 디코딩에서 문제가 있을 수 있어 CPU 사용 또는 float32 필요
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS는 VAE 디코딩을 위해 CPU 사용 (검은색 이미지 문제 해결)
                device = "mps"  # UNet은 MPS 사용
                vae_device = "cpu"  # VAE는 CPU 사용 (검은색 이미지 방지)
                dtype = torch.float16  # MPS는 float16 지원
                print("🍎 Apple Silicon (M1/M2) 감지 - MPS 백엔드 사용 (VAE는 CPU)")
            elif torch.cuda.is_available():
                device = "cuda"
                vae_device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                vae_device = "cpu"
                dtype = torch.float32
                print("⚠️ GPU를 찾을 수 없습니다. CPU 모드로 실행합니다 (느림)")
            
            # M2 맥북을 위한 더 작고 효율적인 모델 선택
            # stable-diffusion-v1-4가 v1-5보다 약간 작고 빠름
            model_name = "CompVis/stable-diffusion-v1-4"
            
            # 모델 로드 (처음 실행 시 다운로드됨, 약 4GB)
            print(f"Stable Diffusion 모델 로드 중... (장치: {device}, 모델: {model_name})")
            print("⏳ 처음 실행 시 모델 다운로드가 필요합니다 (약 4GB, 몇 분 소요)")
            
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    safety_checker=None,  # 속도 향상 및 메모리 절약
                    requires_safety_checker=False
                )
                
                # MPS 백엔드 사용 시 추가 최적화
                if device == "mps":
                    # VAE는 CPU에서 실행 (검은색 이미지 문제 해결)
                    pipe.vae.to(vae_device)
                    # UNet과 text_encoder는 MPS 사용
                    pipe.unet.to(device)
                    pipe.text_encoder.to(device)
                    # 메모리 효율을 위한 설정
                    pipe.enable_attention_slicing()  # 메모리 사용량 감소
                else:
                    pipe = pipe.to(device)
                    
            except Exception as load_error:
                print(f"⚠️ 모델 로드 실패: {load_error}")
                print("💡 다른 모델로 재시도 중...")
                # 대안 모델 시도
                try:
                    model_name = "runwayml/stable-diffusion-v1-5"
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    pipe = pipe.to(device)
                    if device == "mps":
                        pipe.enable_attention_slicing()
                except Exception as e2:
                    print(f"⚠️ 대안 모델도 로드 실패: {e2}")
                    return None
            
            # 이미지 생성 (M2 최적화 설정 - 속도 최우선)
            print(f"이미지 생성 중... 프롬프트: {prompt[:80]}...")
            print(f"⏳ 생성 시간: 약 20-40초 (최적화됨)")
            
            # CPU fallback용 재로드 함수
            def _reload_pipeline_for_cpu():
                """CPU fallback을 위해 float32로 재로드"""
                print("💡 CPU 모드로 파이프라인 재로드 중...")
                return StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # CPU는 float32 필요
                    safety_checker=None,
                    requires_safety_checker=False
                ).to("cpu")
            
            # Negative prompt 추가 (이상한 얼굴 방지 - 강화)
            if negative_prompt is None:
                negative_prompt = "ugly face, distorted face, deformed face, scary face, horror face, ghost face, zombie face, demon face, monster face, alien face, blurry face, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, mutated hands, mutated fingers, deformed, bad anatomy, asymmetrical face, crooked nose, weird eyes, unnatural skin, corpse-like, dead eyes, blurry, watermark, grainy, signature, cut off, draft, low quality, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, bad body, blurry, bad anatomy, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, jpeg artifacts, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
            
            # MPS에서는 VAE 디코딩이 CPU에서 실행되므로 주의
            with torch.no_grad():
                try:
                    result = pipe(
                        prompt,
                        negative_prompt=negative_prompt,  # Negative prompt 추가
                        num_inference_steps=20,  # 속도 최적화 (얼굴이 안 보이므로 20단계로 충분)
                        guidance_scale=7.0,  # 빠른 생성
                        height=512,  # M2 메모리 제한 고려
                        width=512
                    )
                    
                    # 결과 이미지 확인 및 검증
                    image = result.images[0]
                    
                    # 검은색 이미지 체크 (모든 픽셀이 검은색인지 확인)
                    import numpy as np
                    img_array = np.array(image)
                    if np.all(img_array == 0) or np.all(img_array < 10):
                        print("⚠️ 검은색 이미지 감지. CPU 모드로 재생성 중...")
                        # CPU용 파이프라인 재로드 (float32)
                        pipe_cpu = _reload_pipeline_for_cpu()
                        pipe_cpu.enable_attention_slicing()
                        negative_prompt = "ugly face, distorted face, deformed face, scary face, horror face, ghost face, zombie face, demon face, monster face, alien face, blurry face, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, mutated hands, mutated fingers, deformed, bad anatomy, asymmetrical face, crooked nose, weird eyes, unnatural skin, corpse-like, dead eyes, blurry, watermark, grainy, signature, cut off, draft, low quality, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, bad body, blurry, bad anatomy, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, jpeg artifacts, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                        with torch.no_grad():
                            result = pipe_cpu(
                                prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=30,  # 얼굴 품질 향상
                                guidance_scale=8.0,  # 얼굴 품질 향상
                                height=512,
                                width=512
                            )
                        image = result.images[0]
                    
                except RuntimeError as mps_error:
                    # MPS 관련 오류 시 CPU로 fallback
                    if "mps" in str(mps_error).lower() or device == "mps":
                        print("⚠️ MPS 오류 발생. CPU 모드로 재시도 중...")
                        # CPU용 파이프라인 재로드 (float32)
                        pipe_cpu = _reload_pipeline_for_cpu()
                        pipe_cpu.enable_attention_slicing()
                        negative_prompt = "ugly face, distorted face, deformed face, scary face, horror face, ghost face, zombie face, demon face, monster face, alien face, blurry face, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, mutated hands, mutated fingers, deformed, bad anatomy, asymmetrical face, crooked nose, weird eyes, unnatural skin, corpse-like, dead eyes, blurry, watermark, grainy, signature, cut off, draft, low quality, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, bad body, blurry, bad anatomy, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, jpeg artifacts, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                        with torch.no_grad():
                            result = pipe_cpu(
                                prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=30,  # 얼굴 품질 향상
                                guidance_scale=8.0,  # 얼굴 품질 향상
                                height=512,
                                width=512
                            )
                        image = result.images[0]
                    else:
                        raise
            
            print("✅ 이미지 생성 완료!")
            return image
            
        except ImportError:
            print("⚠️ diffusers 라이브러리가 설치되지 않았습니다.")
            print("💡 설치 방법: pip install diffusers accelerate")
            return None
        except Exception as e:
            print(f"⚠️ Stable Diffusion 로컬 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_with_stability_ai(self, prompt: str) -> Optional[Image.Image]:
        """Stability AI API 사용 (유료)"""
        try:
            if not self.api_key:
                print("⚠️ STABILITY_AI_API_KEY가 설정되지 않았습니다.")
                return None
            
            api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                image_base64 = result["artifacts"][0]["base64"]
                import base64
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_bytes))
                return image
            else:
                print(f"⚠️ Stability AI API 오류: {response.status_code}")
                return None
        except Exception as e:
            print(f"⚠️ Stability AI API 호출 실패: {e}")
            return None
    
    def generate_multiple_outfits(self, outfit_descriptions: List[Dict], 
                                 style_info: Dict = None) -> List[Optional[Image.Image]]:
        """여러 코디에 대한 이미지 일괄 생성"""
        images = []
        for desc in outfit_descriptions:
            image = self.generate_outfit_image(desc, style_info)
            images.append(image)
        return images


# 사용 예시
if __name__ == "__main__":
    generator = OutfitImageGenerator(method="huggingface_api")
    
    outfit_desc = {
        "items": ["red shirt", "blue jeans"],
        "style": "캐주얼",
        "colors": ["red", "blue"]
    }
    
    image = generator.generate_outfit_image(outfit_desc)
    if image:
        image.save("generated_outfit.png")
        print("✅ 이미지 생성 완료: generated_outfit.png")

