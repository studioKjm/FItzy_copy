"""
추천 코디 AI 이미지 생성 유틸리티
Stable Diffusion 로컬 실행 (MPS 최적화)
"""

import os
from PIL import Image
from typing import Optional, Dict
import torch
from diffusers import StableDiffusionPipeline


class OutfitImageGenerator:
    """추천 코디 AI 이미지 생성 클래스 - 간소화 버전"""
    
    def __init__(self, method: str = "stable_diffusion"):
        """
        Args:
            method: 이미지 생성 방법 (현재는 "stable_diffusion"만 지원)
        """
        self.method = method
        
        # MPS 사용 가능 여부 확인
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("❌ MPS (GPU)를 사용할 수 없습니다. MacBook M1/M2가 필요합니다!")
        
        # 파이프라인은 나중에 로드 (지연 로딩)
        self.pipe = None
        self.device = "mps"
        self.vae_device = "cpu"
    
    def _load_pipeline(self):
        """파이프라인 지연 로딩"""
        if self.pipe is not None:
            return
        
        print("🍎 Apple Silicon (M1/M2) 감지 - MPS 백엔드 사용 (GPU 가속)")
        print("⚡ Stable Diffusion 2.1 모델 로드 중... (SD 1.4보다 개선된 버전)")
        print("⏳ 처음 실행 시 모델 다운로드가 필요합니다 (약 5GB, 몇 분 소요)")
        
        try:
            # Stable Diffusion 2.1: SD 1.4보다 개선, 메모리 효율적
            print("📦 Stable Diffusion 2.1 시도 중... (SD 1.4보다 개선)")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                device_map=None
            )
            print("✅ Stable Diffusion 2.1 모델 로드 성공!")
        except Exception as e:
            print(f"⚠️ SD 2.1 로드 실패: {e}")
            print("📦 Stable Diffusion v1.4로 폴백...")
            
            # 폴백: SD 1.4
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                device_map=None
            )
            print("✅ SD 1.4 모델 로드 완료")
        
        # 컴포넌트별 장치 배치
        self.pipe.vae = self.pipe.vae.to(self.vae_device, non_blocking=False)
        self.pipe.unet = self.pipe.unet.float().to(self.device, non_blocking=False)
        self.pipe.text_encoder = self.pipe.text_encoder.float().to("cpu", non_blocking=False)
        
        # 최적화 설정
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        
        torch.mps.synchronize()
        print("✅ 모델 로드 완료")
    
    def _build_prompt(self, outfit_description: Dict) -> str:
        """효과적인 프롬프트 생성 (간결하고 명확하게)"""
        items = outfit_description.get("items", [])
        style = outfit_description.get("style", "캐주얼")
        gender = outfit_description.get("gender", "공용")
        
        # 성별 키워드 (간결)
        gender_keyword = "man" if gender == "남성" else "woman" if gender == "여성" else "person"
        
        # 색상/타입 변환
        color_map = {
            "검은색": "black", "흰색": "white", "빨간색": "red", "파란색": "blue",
            "노란색": "yellow", "초록색": "green", "분홍색": "pink", "보라색": "purple",
            "회색": "gray", "갈색": "brown", "베이지": "beige", "카키": "khaki",
            "네이비": "navy", "오렌지": "orange", "파스텔": "pastel"
        }
        
        # 아이템을 최대 2개로 제한 (정확도 향상)
        processed_items = []
        for item in items[:2]:  # 3개 -> 2개로 제한
            # 브랜드명 제거
            for brand in ["유니클로", "리바이스", "컨버스", "나이키", "아디다스", "자라", "H&M", 
                         "uniqlo", "levis", "converse", "nike", "adidas", "zara"]:
                item = item.replace(brand, "").strip()
            
            # 색상 영어 변환
            for kr_color, en_color in color_map.items():
                if kr_color in item:
                    item = item.replace(kr_color, en_color)
            
            # 타입 영어 변환
            item = item.replace("반팔", "short sleeve").replace("긴팔", "long sleeve")
            item = item.replace("티셔츠", "t-shirt").replace("셔츠", "shirt")
            item = item.replace("바지", "pants").replace("반바지", "shorts")
            item = item.replace("재킷", "jacket").replace("가디건", "cardigan")
            item = item.replace("부츠", "boots").replace("스니커즈", "sneakers")
            item = " ".join(item.split())
            
            if item:
                processed_items.append(item)
        
        # 아이템을 간결하게 표현
        if len(processed_items) >= 2:
            items_text = f"{processed_items[0]}, {processed_items[1]}"
        elif len(processed_items) == 1:
            items_text = processed_items[0]
        else:
            items_text = "casual outfit"
        
        # 색상 정확도 최우선 프롬프트 (UPPERCASE 강조)
        if len(processed_items) >= 2:
            # 각 아이템의 색상을 UPPERCASE로 강조
            item1_upper = processed_items[0].upper()
            item2_upper = processed_items[1].upper()
            prompt = (
                f"one single mannequin only, {item1_upper}, "
                f"{item2_upper}, EXACT COLORS, product photo, centered"
            )
        elif len(processed_items) == 1:
            item_upper = processed_items[0].upper()
            prompt = f"single mannequin wearing {item_upper}, EXACT COLOR, product photo"
        else:
            prompt = "single mannequin with clothing"
        
        return prompt
    
    def generate_outfit_image(self, outfit_description: Dict, style_info: Dict = None) -> Optional[Image.Image]:
        """코디 설명을 바탕으로 AI 이미지 생성"""
        try:
            self._load_pipeline()
            
            prompt = self._build_prompt(outfit_description)
            # 매우 강력한 negative prompt (얼굴 제거 + 여러 마네킹 제거 + 색상 혼동 방지)
            negative_prompt = (
                "face, head, eyes, nose, mouth, lips, hair, neck, portrait, person, human face, "
                "multiple people, multiple mannequins, two mannequins, crowd, group, "
                "wrong colors, incorrect colors, color swap, reversed colors, "
                "white pants, shorts, blurry, low quality"
            )
            
            print(f"이미지 생성 중... 프롬프트: {prompt[:80]}...")
            print(f"⏳ 생성 시간: 약 15-30초")
            
            # 패치: prepare_latents를 MPS에서 생성하도록 수정
            import types
            original_prepare_latents = self.pipe.prepare_latents
            
            def patched_prepare_latents(self_pipe, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
                if latents is None:
                    latents = original_prepare_latents(
                        batch_size, num_channels_latents, height, width, dtype, self.device, generator, None
                    )
                    # 이미 MPS로 생성되지만 확인
                    if isinstance(latents, torch.Tensor) and latents.device.type != self.device:
                        latents = latents.to(self.device, non_blocking=False)
                else:
                    # latents가 제공된 경우에도 MPS로 이동
                    if isinstance(latents, torch.Tensor) and latents.device.type != self.device:
                        latents = latents.to(self.device, non_blocking=False)
                return latents
            
            # UNet forward 패치: encoder_hidden_states를 MPS로 강제 이동
            original_unet_forward = self.pipe.unet.forward
            
            def patched_unet_forward(self_unet, sample, timestep, encoder_hidden_states=None, timestep_cond=None, **kwargs):
                # 모든 입력 텐서를 MPS로 이동
                if isinstance(sample, torch.Tensor) and sample.device.type != self.device:
                    sample = sample.to(self.device, non_blocking=False)
                
                if isinstance(timestep, torch.Tensor) and timestep.device.type != self.device:
                    timestep = timestep.to(self.device, non_blocking=False)
                elif not isinstance(timestep, torch.Tensor):
                    timestep = torch.tensor([timestep], device=self.device, dtype=torch.long)
                
                # encoder_hidden_states는 반드시 MPS로 이동
                if encoder_hidden_states is not None:
                    if isinstance(encoder_hidden_states, torch.Tensor):
                        if encoder_hidden_states.device.type != self.device:
                            encoder_hidden_states = encoder_hidden_states.to(self.device, non_blocking=False)
                    elif isinstance(encoder_hidden_states, (list, tuple)):
                        encoder_hidden_states = tuple(
                            h.to(self.device, non_blocking=False) if isinstance(h, torch.Tensor) and h.device.type != self.device else h 
                            for h in encoder_hidden_states
                        )
                
                if timestep_cond is not None and isinstance(timestep_cond, torch.Tensor) and timestep_cond.device.type != self.device:
                    timestep_cond = timestep_cond.to(self.device, non_blocking=False)
                
                # kwargs의 텐서도 MPS로 이동
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.device.type != self.device:
                        kwargs[key] = value.to(self.device, non_blocking=False)
                    elif isinstance(value, (list, tuple)):
                        kwargs[key] = type(value)(
                            v.to(self.device, non_blocking=False) if isinstance(v, torch.Tensor) and v.device.type != self.device else v 
                            for v in value
                        )
                
                return original_unet_forward(sample, timestep, encoder_hidden_states, timestep_cond, **kwargs)
            
            # 스케줄러 step 패치: 모든 텐서를 MPS로 이동
            original_scheduler_step = self.pipe.scheduler.step
            
            def patched_scheduler_step(self_scheduler, model_output, timestep, sample, return_dict=True, **kwargs):
                # 모든 입력 텐서를 MPS로 이동
                if isinstance(model_output, torch.Tensor) and model_output.device.type != self.device:
                    model_output = model_output.to(self.device, non_blocking=False)
                if isinstance(sample, torch.Tensor) and sample.device.type != self.device:
                    sample = sample.to(self.device, non_blocking=False)
                if isinstance(timestep, torch.Tensor) and timestep.device.type != self.device:
                    timestep = timestep.to(self.device, non_blocking=False)
                elif not isinstance(timestep, torch.Tensor):
                    timestep = torch.tensor([timestep], device=self.device, dtype=torch.long)
                
                result = original_scheduler_step(model_output, timestep, sample, return_dict=return_dict, **kwargs)
                
                # 결과 텐서도 MPS로 이동
                if isinstance(result, tuple):
                    result = tuple(
                        r.to(self.device, non_blocking=False) if isinstance(r, torch.Tensor) and r.device.type != self.device else r
                        for r in result
                    )
                elif isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor) and value.device.type != self.device:
                            result[key] = value.to(self.device, non_blocking=False)
                
                return result
            
            # VAE decode 패치: latents를 CPU로 이동 (VAE는 CPU에서 실행)
            original_vae_decode = self.pipe.vae.decode
            
            def patched_vae_decode(self_vae, z, return_dict=True, **kwargs):
                # z(latents)가 MPS에 있으면 CPU로 이동
                if isinstance(z, torch.Tensor) and z.device.type == "mps":
                    z = z.to("cpu", non_blocking=False)
                return original_vae_decode(z, return_dict=return_dict, **kwargs)
            
            # 패치 적용
            self.pipe.prepare_latents = types.MethodType(patched_prepare_latents, self.pipe)
            self.pipe.unet.forward = types.MethodType(patched_unet_forward, self.pipe.unet)
            self.pipe.scheduler.step = types.MethodType(patched_scheduler_step, self.pipe.scheduler)
            self.pipe.vae.decode = types.MethodType(patched_vae_decode, self.pipe.vae)
            
            try:
                with torch.no_grad():
                    # encode_prompt를 먼저 호출하여 prompt_embeds 생성
                    prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                        prompt=prompt,
                        device=torch.device("cpu"),  # TextEncoder는 CPU
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt
                    )
                    
                    # prompt_embeds를 MPS로 이동
                    if isinstance(prompt_embeds, torch.Tensor):
                        prompt_embeds = prompt_embeds.to(self.device, non_blocking=False)
                    if isinstance(negative_prompt_embeds, torch.Tensor):
                        negative_prompt_embeds = negative_prompt_embeds.to(self.device, non_blocking=False)
                    
                    # pipe() 호출 시 prompt_embeds 사용
                    # SDXL-Turbo는 1-4 steps, guidance_scale 0.0 권장
                    # SD 1.4는 기존 설정 유지
                    seed = 42
                    generator_obj = torch.Generator(device="cpu").manual_seed(seed)
                    
                    # SDXL-Turbo 감지 (모델명으로 판단)
                    is_sdxl_turbo = "sdxl-turbo" in str(self.pipe.config._name_or_path).lower()
                    
                    if is_sdxl_turbo:
                        # SDXL-Turbo: 1 step, no guidance (빠르고 정확)
                        result = self.pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            num_inference_steps=4,  # Turbo는 1-4 steps
                            guidance_scale=0.0,  # Turbo는 guidance 불필요
                            height=512,
                            width=512,
                            generator=generator_obj
                        )
                    else:
                        # SD 1.4: 기존 설정
                        result = self.pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            num_inference_steps=30,
                            guidance_scale=15.0,
                            height=512,
                            width=512,
                            generator=generator_obj
                        )
                
                image = result.images[0]
                
                # 후처리: 상단 40% 크롭하여 얼굴/목 완전 제거
                width, height = image.size
                crop_top = int(height * 0.40)  # 상단 40% 제거 (목까지 제거)
                cropped_image = image.crop((0, crop_top, width, height))
                
                # 원래 크기로 조정 (흰색 여백 추가)
                from PIL import Image as PILImage
                final_image = PILImage.new('RGB', (width, height), color=(255, 255, 255))
                final_image.paste(cropped_image, (0, 0))
                
                print("✅ 이미지 생성 완료 (얼굴/목 완전 제거)")
                return final_image
            finally:
                # 패치 복원
                self.pipe.prepare_latents = original_prepare_latents
                self.pipe.unet.forward = original_unet_forward
                self.pipe.scheduler.step = original_scheduler_step
                self.pipe.vae.decode = original_vae_decode
            
        except Exception as e:
            print(f"⚠️ 이미지 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
