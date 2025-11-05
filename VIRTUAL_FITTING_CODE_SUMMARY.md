# 의상 피팅 관련 소스코드 요약

## 📋 목차
1. [VirtualFittingSystem 클래스 전체](#1-virtualfittingsystem-클래스-전체)
2. [app.py에서의 사용 부분](#2-apppy에서의-사용-부분)
3. [핵심 메서드 설명](#3-핵심-메서드-설명)
4. [최신 개선사항](#4-최신-개선사항)

---

## 1. VirtualFittingSystem 클래스 전체

**파일 위치**: `src/utils/virtual_fitting.py`

```python
"""
가상 피팅 시스템 - 업로드된 이미지에 추천 코디 합성
YOLO 탐지 → 아이템별 생성 → 영역 합성 → 색상 보정
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Tuple, Optional
from diffusers import StableDiffusionInpaintPipeline


class VirtualFittingSystem:
    """가상 피팅 시스템 - 사용자 이미지에 추천 코디 합성"""
    
    def __init__(self, yolo_detector, clip_analyzer):
        """
        Args:
            yolo_detector: YOLODetector 인스턴스
            clip_analyzer: CLIPAnalyzer 인스턴스
        """
        self.yolo_detector = yolo_detector
        self.clip_analyzer = clip_analyzer
        self.inpaint_pipe = None  # inpainting 파이프라인 (필요 시 로드)
        
        # MPS (GPU) 사용 가능 여부 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            self.vae_device = "cpu"
            print("🍎 MPS (GPU) 사용 가능 - 빠른 이미지 생성")
        else:
            self.device = "cpu"
            self.vae_device = "cpu"
            print("⚠️ MPS 사용 불가 - CPU 모드로 실행")
    
    def detect_clothing_regions(self, image: Image.Image) -> Dict:
        """
        YOLO로 의류 영역 탐지
        
        Returns:
            {
                "top": {"bbox": [x1, y1, x2, y2], "class": "...", "confidence": 0.9},
                "bottom": {"bbox": [...], ...},
                "person": {"bbox": [...], ...}
            }
        """
        # YOLO 탐지 실행
        result = self.yolo_detector.detect_clothes(image)
        items = result.get("items", [])
        
        regions = {}
        
        # 탐지된 아이템을 상의/하의/전신으로 분류
        for item in items:
            class_name = item.get("class", "").lower()
            class_en = item.get("class_en", "").lower()
            bbox = item.get("bbox", [])
            
            if not bbox or len(bbox) != 4:
                continue
            
            # 상의 분류
            if any(keyword in class_name or keyword in class_en 
                   for keyword in ["상의", "top", "shirt", "t-shirt", "jacket", "outwear"]):
                if "top" not in regions or item.get("confidence", 0) > regions["top"].get("confidence", 0):
                    regions["top"] = {
                        "bbox": bbox,
                        "class": item.get("class", ""),
                        "confidence": item.get("confidence", 0)
                    }
            
            # 하의 분류
            elif any(keyword in class_name or keyword in class_en 
                     for keyword in ["하의", "bottom", "pants", "바지", "skirt", "치마"]):
                if "bottom" not in regions or item.get("confidence", 0) > regions["bottom"].get("confidence", 0):
                    regions["bottom"] = {
                        "bbox": bbox,
                        "class": item.get("class", ""),
                        "confidence": item.get("confidence", 0)
                    }
            
            # 전신 (person)
            elif "person" in class_name or "person" in class_en:
                if "person" not in regions or item.get("confidence", 0) > regions["person"].get("confidence", 0):
                    regions["person"] = {
                        "bbox": bbox,
                        "class": item.get("class", ""),
                        "confidence": item.get("confidence", 0)
                    }
        
        return regions
    
    def composite_outfit_on_image(self, original_image: Image.Image, 
                                 outfit_items: List[str],
                                 gender: str = "남성") -> Optional[Image.Image]:
        """
        원본 이미지에 추천 코디를 합성 (핵심 메서드)
        
        Args:
            original_image: 사용자 업로드 이미지
            outfit_items: 추천 코디 아이템 리스트 (예: ["빨간색 긴팔 셔츠", "검은색 바지"])
            gender: 성별
        
        Returns:
            합성된 이미지 또는 None
        """
        try:
            print("🎨 가상 피팅 시작...")
            print(f"   - 아이템: {outfit_items}")
            print(f"   - 성별: {gender}")
            
            # 1. 의류 영역 탐지
            regions = self.detect_clothing_regions(original_image)
            
            print(f"   - 탐지된 영역: {list(regions.keys())}")
            
            if not regions:
                print("⚠️ 의류 영역을 찾을 수 없습니다.")
                return self._create_text_overlay_image(original_image, outfit_items)
            
            # 2. OpenCV 형식으로 변환
            img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]
            
            # 3. Inpainting으로 실제 의류 합성
            self._load_inpaint_pipeline()
            
            if self.inpaint_pipe is None:
                print("⚠️ Inpainting 모델 없음. 간단한 색상 오버레이 사용")
                return self._simple_color_overlay(img_cv, regions, outfit_items, width, height)
            
            # Inpainting으로 각 아이템 합성 (상의 + 하의 모두 처리)
            result_pil = original_image.copy()
            
            # 상의와 하의 모두 처리 (최대 2개)
            for idx, item in enumerate(outfit_items[:2]):  # 상의 + 하의
                region_type = "top" if idx == 0 else "bottom"
                
                if region_type not in regions:
                    print(f"⚠️ {region_type} 영역 없음, 다음 아이템으로")
                    continue
                
                bbox = regions[region_type]["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # 마스크 생성 (Inpainting용)
                mask_pil = Image.new("L", (width, height), 0)  # 검은색
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask_pil)
                draw.rectangle([x1, y1, x2, y2], fill=255)  # 흰색 = 교체할 영역
                
                # 프롬프트 생성 (region_type 전달!)
                prompt = self._build_inpaint_prompt(item, gender, region_type)
                
                # 성별에 따른 negative prompt 강화
                if gender == "남성":
                    negative_prompt = (
                        "woman, female, women's clothing, women's shoes, high heels, "
                        "breasts, cleavage, feminine curves, "
                        "wrong color, mismatched clothes, double clothing, overlay, blur, "
                        "distorted body, unrealistic fabric, old outfit, wrong gender clothing, "
                        "face, head, portrait, drawing, painting, illustration, cartoon, "
                        "anime, unrealistic, fake, artificial, CGI, 3D render, computer graphics"
                    )
                else:  # 여성
                    negative_prompt = (
                        "man, male, men's clothing, men's shoes, "
                        "wrong color, mismatched clothes, double clothing, overlay, blur, "
                        "distorted body, unrealistic fabric, old outfit, wrong gender clothing, "
                        "face, head, portrait, drawing, painting, illustration, cartoon, "
                        "anime, unrealistic, fake, artificial, CGI, 3D render, computer graphics"
                    )
                
                print(f"🎨 {region_type} 영역 Inpainting 중...")
                print(f"   - 프롬프트: {prompt}")
                
                try:
                    # 이미지와 마스크를 최적 크기로 리사이즈 (한 번만, 속도 향상)
                    # 원본 크기에 비례하여 리사이즈 (너무 크면 느림)
                    max_size = 512
                    orig_w, orig_h = original_image.size
                    
                    # 리사이즈 필요 여부 확인
                    needs_resize = max(orig_w, orig_h) > max_size
                    
                    if needs_resize:
                        ratio = max_size / max(orig_w, orig_h)
                        target_size = (int(orig_w * ratio), int(orig_h * ratio))
                        # 한 번만 리사이즈
                        result_pil_for_inpaint = result_pil.resize(target_size, Image.Resampling.LANCZOS)
                        mask_pil_for_inpaint = mask_pil.resize(target_size, Image.Resampling.LANCZOS)
                        print(f"   - 이미지 리사이즈: {original_image.size} → {target_size}")
                    else:
                        # 리사이즈 불필요
                        result_pil_for_inpaint = result_pil
                        mask_pil_for_inpaint = mask_pil
                        print(f"   - 원본 크기 사용: {original_image.size}")
                    
                    # Inpainting 실행 (GPU/CPU 모드, 자연스러운 합성)
                    with torch.no_grad():
                        try:
                            result = self.inpaint_pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                image=result_pil_for_inpaint,
                                mask_image=mask_pil_for_inpaint,
                                num_inference_steps=20 if self.device == "mps" else 10,  # GPU: 더 많은 steps, CPU: 빠르게
                                guidance_scale=9.0,  # 프롬프트 준수도 매우 높임
                                strength=0.9  # 90% 변경 (더 강하게)
                            )
                        except (RuntimeError, TypeError) as e:
                            error_str = str(e)
                            if "unexpected keyword argument" in error_str and "generator" in error_str:
                                # VAE decode 시그니처 오류 - 패치 재적용 및 재시도
                                print(f"   ⚠️ VAE decode 시그니처 오류, 패치 재적용 중...")
                                # VAE decode 패치 재적용
                                original_decode = self.inpaint_pipe.vae.decode
                                def patched_vae_decode_fix(self_vae, z, return_dict=True, **kwargs):
                                    if z.device.type != "cpu":
                                        z = z.to("cpu", non_blocking=False)
                                    # generator 인자 제거
                                    kwargs.pop('generator', None)
                                    return original_decode(z, return_dict=return_dict, **kwargs)
                                self.inpaint_pipe.vae.decode = patched_vae_decode_fix.__get__(self.inpaint_pipe.vae, type(self.inpaint_pipe.vae))
                                # 재시도
                                result = self.inpaint_pipe(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    image=result_pil_for_inpaint,
                                    mask_image=mask_pil_for_inpaint,
                                    num_inference_steps=20 if self.device == "mps" else 10,
                                    guidance_scale=9.0,
                                    strength=0.9
                                )
                            elif "must be on the same device" in error_str or "same device" in error_str:
                                # 디바이스 오류 - MPS 패치 재적용
                                print(f"   ⚠️ 디바이스 오류, MPS 패치 재적용 중...")
                                # 패치 재적용
                                self._apply_mps_patches()
                                # 재시도
                                result = self.inpaint_pipe(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    image=result_pil_for_inpaint,
                                    mask_image=mask_pil_for_inpaint,
                                    num_inference_steps=20 if self.device == "mps" else 10,
                                    guidance_scale=9.0,
                                    strength=0.9
                                )
                            else:
                                # 다른 오류는 재발생
                                print(f"   ❌ 예상치 못한 오류: {error_str[:100]}")
                                raise
                    
                    # 결과를 원본 크기로 복원
                    generated = result.images[0]
                    
                    # 리사이즈된 경우에만 원본 크기로 복원 (한 번만)
                    if needs_resize and generated.size != original_image.size:
                        generated = generated.resize(original_image.size, Image.Resampling.LANCZOS)
                        mask_pil_full = mask_pil.resize(original_image.size, Image.Resampling.LANCZOS)
                    else:
                        # 리사이즈하지 않은 경우 마스크도 그대로 사용
                        mask_pil_full = mask_pil
                    
                    # 마스크 영역만 합성 (나머지는 원본 유지)
                    result_np = np.array(result_pil)
                    generated_np = np.array(generated)
                    
                    mask_np = np.array(mask_pil_full) > 127  # 이진 마스크
                    mask_3d = np.stack([mask_np] * 3, axis=2).astype(float)  # 0.0 또는 1.0
                    
                    # 마스크 영역은 생성된 이미지, 나머지는 원본
                    blended = result_np.astype(float) * (1.0 - mask_3d) + generated_np.astype(float) * mask_3d
                    result_np = np.clip(blended, 0, 255).astype(np.uint8)
                    
                    result_pil = Image.fromarray(result_np)
                    
                    print(f"✅ {region_type} 영역 Inpainting 완료 (실제 합성됨)")
                    print(f"   - 마스크 영역 크기: {np.sum(mask_np)} 픽셀")
                    
                except Exception as e:
                    print(f"⚠️ Inpainting 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    return self._simple_color_overlay(img_cv, regions, outfit_items, width, height)
            
            print("✅ 가상 피팅 완료 (Inpainting)")
            return result_pil
            
        except Exception as e:
            print(f"⚠️ 가상 피팅 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_inpaint_pipeline(self):
        """Stable Diffusion Inpainting 파이프라인 로드"""
        if self.inpaint_pipe is not None:
            return
        
        print("🎨 Stable Diffusion Inpainting 모델 로드 중...")
        print(f"   - 장치: {self.device.upper()} 모드")
        
        try:
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float32,
                safety_checker=None,
                device_map=None
            )
            
            # 디바이스 배치 (MPS: UNet만, CPU: VAE/TextEncoder)
            if self.device == "mps":
                self.inpaint_pipe.unet = self.inpaint_pipe.unet.float().to(self.device, non_blocking=False)
                self.inpaint_pipe.vae = self.inpaint_pipe.vae.to(self.vae_device, non_blocking=False)
                self.inpaint_pipe.text_encoder = self.inpaint_pipe.text_encoder.float().to("cpu", non_blocking=False)
                
                # MPS 패치 적용
                self._patch_vae_for_mps()
                self._apply_mps_patches()
                
                print("✅ Inpainting 모델 로드 완료 (MPS/GPU 모드)")
            else:
                self.inpaint_pipe.unet = self.inpaint_pipe.unet.to("cpu")
                self.inpaint_pipe.vae = self.inpaint_pipe.vae.to("cpu")
                self.inpaint_pipe.text_encoder = self.inpaint_pipe.text_encoder.to("cpu")
                print("✅ Inpainting 모델 로드 완료 (CPU 모드)")
        except Exception as e:
            print(f"⚠️ Inpainting 모델 로드 실패: {e}")
            self.inpaint_pipe = None
    
    def _patch_vae_for_mps(self):
        """VAE의 encode/decode 메서드를 패치하여 MPS와 호환되도록"""
        if self.device != "mps":
            return
        
        # VAE encode 패치
        original_encode = self.inpaint_pipe.vae.encode
        
        def patched_vae_encode(self_vae, x, return_dict=True, **kwargs):
            if x.device.type != "cpu":
                x = x.to("cpu", non_blocking=False)
            result = original_encode(x, return_dict=return_dict, **kwargs)
            if return_dict:
                if hasattr(result, 'latent_dist'):
                    pass
                return result
            else:
                if isinstance(result, tuple):
                    return tuple(r.to(self.device, non_blocking=False) if isinstance(r, torch.Tensor) and r.device.type != self.device else r for r in result)
                return result.to(self.device, non_blocking=False) if isinstance(result, torch.Tensor) and result.device.type != self.device else result
        
        self.inpaint_pipe.vae.encode = patched_vae_encode.__get__(self.inpaint_pipe.vae, type(self.inpaint_pipe.vae))
        
        # VAE decode 패치 - generator 인자 명시적으로 처리
        original_decode = self.inpaint_pipe.vae.decode
        
        import inspect
        sig = inspect.signature(original_decode)
        print(f"   📋 VAE decode 원본 시그니처: {sig}")
        
        def patched_vae_decode(self_vae, z, return_dict=True, generator=None, **kwargs):
            if z.device.type != "cpu":
                z = z.to("cpu", non_blocking=False)
            kwargs.pop('generator', None)
            return original_decode(z, return_dict=return_dict, **kwargs)
        
        self.inpaint_pipe.vae.decode = patched_vae_decode.__get__(self.inpaint_pipe.vae, type(self.inpaint_pipe.vae))
        
        print("   ✅ VAE encode/decode 패치 적용 완료")
    
    def _apply_mps_patches(self):
        """MPS 디바이스 불일치 오류 방지를 위한 패치 적용"""
        if self.device != "mps":
            return
        
        # UNet forward 패치
        original_unet_forward = self.inpaint_pipe.unet.forward
        
        def patched_unet_forward(self_unet, sample, timestep, encoder_hidden_states=None, **kwargs):
            if sample.device.type != self.device:
                sample = sample.to(self.device, non_blocking=False)
            if isinstance(timestep, torch.Tensor) and timestep.device.type != self.device:
                timestep = timestep.to(self.device, non_blocking=False)
            if encoder_hidden_states is not None and encoder_hidden_states.device.type != self.device:
                encoder_hidden_states = encoder_hidden_states.to(self.device, non_blocking=False)
            
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.device.type != self.device:
                    kwargs[key] = value.to(self.device, non_blocking=False)
            
            return original_unet_forward(sample, timestep, encoder_hidden_states, **kwargs)
        
        self.inpaint_pipe.unet.forward = patched_unet_forward.__get__(self.inpaint_pipe.unet, type(self.inpaint_pipe.unet))
        
        # Scheduler step 패치
        original_scheduler_step = self.inpaint_pipe.scheduler.step
        
        def patched_scheduler_step(self_scheduler, model_output, timestep, sample, **kwargs):
            if model_output.device.type != self.device:
                model_output = model_output.to(self.device, non_blocking=False)
            if isinstance(timestep, torch.Tensor) and timestep.device.type != self.device:
                timestep = timestep.to(self.device, non_blocking=False)
            if sample.device.type != self.device:
                sample = sample.to(self.device, non_blocking=False)
            
            return original_scheduler_step(model_output, timestep, sample, **kwargs)
        
        self.inpaint_pipe.scheduler.step = patched_scheduler_step.__get__(self.inpaint_pipe.scheduler, type(self.inpaint_pipe.scheduler))
        
        # prepare_mask_latents 패치 (올바른 시그니처)
        import types
        
        if hasattr(self.inpaint_pipe, 'prepare_mask_latents'):
            original_prepare_mask_latents = self.inpaint_pipe.prepare_mask_latents
            
            def patched_prepare_mask_latents(self_pipe, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance):
                # device를 MPS로 강제
                device = torch.device(self.device)
                # 원본 호출
                mask_latents, masked_image_latents = original_prepare_mask_latents(
                    mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
                )
                # 결과를 MPS로 이동
                if mask_latents.device.type != self.device:
                    mask_latents = mask_latents.to(self.device, non_blocking=False)
                if masked_image_latents.device.type != self.device:
                    masked_image_latents = masked_image_latents.to(self.device, non_blocking=False)
                return mask_latents, masked_image_latents
            
            self.inpaint_pipe.prepare_mask_latents = types.MethodType(patched_prepare_mask_latents, self.inpaint_pipe)
        
        # prepare_latents 패치
        if hasattr(self.inpaint_pipe, 'prepare_latents'):
            original_prepare_latents = self.inpaint_pipe.prepare_latents
            
            def patched_prepare_latents(self_pipe, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, image=None, timestep=None, is_strength_max=True, return_noise=False, return_image_latents=False):
                # device를 MPS로 강제
                device = torch.device(self.device)
                result = original_prepare_latents(
                    batch_size, num_channels_latents, height, width, dtype, device, generator, 
                    latents, image, timestep, is_strength_max, return_noise, return_image_latents
                )
                # 결과를 MPS로 이동
                if isinstance(result, tuple):
                    result = tuple(r.to(self.device, non_blocking=False) if isinstance(r, torch.Tensor) and r.device.type != self.device else r for r in result)
                elif isinstance(result, torch.Tensor) and result.device.type != self.device:
                    result = result.to(self.device, non_blocking=False)
                return result
            
            self.inpaint_pipe.prepare_latents = types.MethodType(patched_prepare_latents, self.inpaint_pipe)
        
        print("   ✅ MPS 패치 적용 완료")
    
    def _build_inpaint_prompt(self, item_text: str, gender: str, region_type: str = "top") -> str:
        """
        Inpainting용 프롬프트 생성 (구체적이고 시각적인 지시문)
        
        Args:
            item_text: 아이템 설명 (예: "빨간색 긴팔 셔츠")
            gender: 성별 ("남성" 또는 "여성")
            region_type: "top" 또는 "bottom"
        
        Returns:
            Inpainting 프롬프트
        """
        # 색상/타입 영어 변환
        color_map = {
            "검은색": "black", "검정": "black", "흰색": "white", "하얀색": "white",
            "빨간색": "red", "빨강": "red", "파란색": "blue", "파랑": "blue",
            "노란색": "yellow", "노랑": "yellow", "초록색": "green", "초록": "green",
            "분홍색": "pink", "분홍": "pink", "보라색": "purple", "보라": "purple",
            "회색": "gray", "회색톤": "gray", "갈색": "brown", "베이지": "beige",
            "카키": "khaki", "네이비": "navy", "오렌지": "orange", "파스텔": "pastel"
        }
        
        # 의류 타입 및 재질 변환
        item_map = {
            "반팔": "short sleeve", "긴팔": "long sleeve",
            "티셔츠": "t-shirt", "티": "t-shirt", "셔츠": "shirt",
            "바지": "pants", "팬츠": "pants", "반바지": "shorts",
            "재킷": "jacket", "자켓": "jacket", "가디건": "cardigan",
            "코트": "coat", "트렌치코트": "trench coat",
            "청바지": "jeans", "진": "jeans",
            "스니커즈": "sneakers", "스니커": "sneakers",
            "부츠": "boots", "신발": "shoes",
            "선글라스": "sunglasses", "안경": "glasses",
            "린넨": "linen", "면": "cotton", "울": "wool",
            "니트": "knit", "스웨터": "sweater"
        }
        
        # 재질 추출
        fabric_map = {
            "면": "cotton", "린넨": "linen", "울": "wool", "니트": "knit",
            "데님": "denim", "청": "denim", "가죽": "leather", "실크": "silk"
        }
        
        # 변환
        en_item = item_text
        
        # 색상 추출
        extracted_colors = []
        for kr, en in color_map.items():
            if kr in item_text:
                extracted_colors.append(en)
                en_item = en_item.replace(kr, en)
        
        extracted_color = extracted_colors[0] if extracted_colors else None
        
        # 의류 타입 추출
        extracted_type = None
        for kr, en in item_map.items():
            if kr in item_text:
                extracted_type = en
                en_item = en_item.replace(kr, en)
        
        # 재질 추출
        extracted_fabric = None
        for kr, en in fabric_map.items():
            if kr in item_text:
                extracted_fabric = en
                break
        
        # 남은 한글 단어 제거
        import re
        en_item = re.sub(r'[가-힣]+', '', en_item).strip()
        en_item = re.sub(r'\s+', ' ', en_item).strip()
        en_item = re.sub(r'\s*(또는|or)\s*.*', '', en_item, flags=re.IGNORECASE).strip()
        
        # 성별 명확히 지정
        gender_kw = "man" if gender == "남성" else "woman" if gender == "여성" else "person"
        
        # 구체적이고 시각적인 프롬프트 생성 (색상과 타입 정확히 명시)
        if region_type == "top":
            # 상의
            if extracted_type and extracted_color:
                fabric_part = f"{extracted_fabric} fabric" if extracted_fabric else "cotton fabric"
                # 타입 정확히 지정
                if "long sleeve" in extracted_type or "긴팔" in item_text:
                    type_spec = "long sleeve shirt"
                elif "short sleeve" in extracted_type or "반팔" in item_text:
                    type_spec = "short sleeve t-shirt"
                else:
                    type_spec = "shirt"
                
                prompt = (
                    f"a {gender_kw} wearing a {extracted_color} {type_spec}, "
                    f"EXACTLY {extracted_color} color, {fabric_part}, "
                    f"realistic fit, naturally worn, proper draping, natural folds, "
                    f"realistic lighting, natural shadows, high quality photo, "
                    f"professional photography, authentic clothing texture"
                )
            elif extracted_type:
                fabric_part = f"{extracted_fabric} fabric" if extracted_fabric else "cotton fabric"
                if "long sleeve" in extracted_type or "긴팔" in item_text:
                    type_spec = "long sleeve shirt"
                elif "short sleeve" in extracted_type or "반팔" in item_text:
                    type_spec = "short sleeve t-shirt"
                else:
                    type_spec = "shirt"
                
                prompt = (
                    f"a {gender_kw} wearing {type_spec}, "
                    f"{fabric_part}, "
                    f"realistic fit, naturally worn, proper draping, natural folds, "
                    f"realistic lighting, natural shadows, high quality photo, "
                    f"professional photography, authentic clothing texture"
                )
            else:
                prompt = (
                    f"a {gender_kw} wearing upper body clothing, "
                    f"realistic fit, naturally worn, proper draping, natural folds, "
                    f"realistic lighting, natural shadows, high quality photo, "
                    f"professional photography, authentic clothing texture"
                )
        else:
            # 하의
            if extracted_type and extracted_color:
                fabric_part = f"{extracted_fabric} fabric" if extracted_fabric else "cotton fabric"
                # 타입 정확히 지정
                if "pants" in extracted_type or "바지" in item_text:
                    type_spec = "slim-fit trousers"
                elif "shorts" in extracted_type or "반바지" in item_text:
                    type_spec = "shorts"
                else:
                    type_spec = "pants"
                
                prompt = (
                    f"a {gender_kw} wearing {extracted_color} {type_spec}, "
                    f"EXACTLY {extracted_color} color, {fabric_part}, "
                    f"realistic fit, naturally worn, proper draping, natural folds, "
                    f"realistic lighting, natural shadows, high quality photo, "
                    f"professional photography, authentic clothing texture"
                )
            elif extracted_type:
                fabric_part = f"{extracted_fabric} fabric" if extracted_fabric else "cotton fabric"
                if "pants" in extracted_type or "바지" in item_text:
                    type_spec = "slim-fit trousers"
                elif "shorts" in extracted_type or "반바지" in item_text:
                    type_spec = "shorts"
                else:
                    type_spec = "pants"
                
                prompt = (
                    f"a {gender_kw} wearing {type_spec}, "
                    f"{fabric_part}, "
                    f"realistic fit, naturally worn, proper draping, natural folds, "
                    f"realistic lighting, natural shadows, high quality photo, "
                    f"professional photography, authentic clothing texture"
                )
            else:
                prompt = (
                    f"a {gender_kw} wearing lower body clothing, "
                    f"realistic fit, naturally worn, proper draping, natural folds, "
                    f"realistic lighting, natural shadows, high quality photo, "
                    f"professional photography, authentic clothing texture"
                )
        
        return prompt
    
    def _simple_color_overlay(self, img_cv: np.ndarray, regions: Dict, 
                             outfit_items: List[str], width: int, height: int) -> Image.Image:
        """폴백: 간단한 색상 오버레이 (Inpainting 실패 시)"""
        result_img = img_cv.copy()
        
        for idx, item in enumerate(outfit_items[:2]):
            region_type = "top" if idx == 0 else "bottom"
            
            if region_type not in regions:
                continue
            
            bbox = regions[region_type]["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            color_bgr = self._extract_target_color(item)
            
            if color_bgr is not None:
                roi = result_img[y1:y2, x1:x2].copy()
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                colored_roi = np.full_like(roi, color_bgr, dtype=np.uint8)
                
                for c in range(3):
                    colored_roi[:, :, c] = np.clip(
                        colored_roi[:, :, c] * (roi_gray.astype(float) / 128.0),
                        0, 255
                    ).astype(np.uint8)
                
                alpha = 0.8
                blended_roi = cv2.addWeighted(colored_roi, alpha, roi, 1-alpha, 0)
                result_img[y1:y2, x1:x2] = blended_roi
                
                print(f"✅ {region_type} 영역 색상 오버레이 적용")
        
        return Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    
    def _extract_target_color(self, item_text: str) -> Optional[Tuple[int, int, int]]:
        """아이템 텍스트에서 목표 색상 추출 (BGR)"""
        color_map_bgr = {
            "검은색": (0, 0, 0),
            "흰색": (255, 255, 255),
            "빨간색": (0, 0, 255),
            "파란색": (255, 0, 0),
            "노란색": (0, 255, 255),
            "초록색": (0, 255, 0),
            "회색": (128, 128, 128),
            "갈색": (42, 42, 165),
            "베이지": (220, 245, 245),
            "네이비": (128, 0, 0),
            "분홍색": (203, 192, 255),
        }
        
        for color_name, bgr in color_map_bgr.items():
            if color_name in item_text:
                return bgr
        
        return None
    
    def _create_text_overlay_image(self, image: Image.Image, items: List[str]) -> Image.Image:
        """의류 탐지 실패 시 원본 이미지에 텍스트 오버레이"""
        from PIL import ImageDraw
        
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        text_lines = ["추천 코디:"] + items
        y_offset = 20
        
        for line in text_lines:
            text_bbox = draw.textbbox((10, y_offset), line)
            draw.rectangle(
                [(text_bbox[0]-5, text_bbox[1]-5), (text_bbox[2]+5, text_bbox[3]+5)], 
                fill=(255, 255, 255)
            )
            draw.text((10, y_offset), line, fill=(0, 0, 0))
            y_offset += 25
        
        return img_with_text
```

---

## 2. app.py에서의 사용 부분

**파일 위치**: `app.py`

### 2.1 초기화 부분

```python
from src.utils.virtual_fitting import VirtualFittingSystem

# 세션 상태에 가상 피팅 시스템 초기화
if 'virtual_fitting' not in st.session_state:
    st.session_state.virtual_fitting = VirtualFittingSystem(
        st.session_state.fashion_recommender.detector,
        st.session_state.fashion_recommender.analyzer
    )
```

### 2.2 통합 추천 생성 부분

```python
# 통합 추천 생성 (성별 + MBTI + 이미지 분석 + 온도/계절 → 스타일 → 아이템 → 제품)
unified_recommendations = st.session_state.recommendation_engine.generate_unified_outfit_recommendations(
    gender, mbti, temp, weather, season,
    detected_items=detected_items_data.get("items", []),
    style_analysis=style_analysis_data
)

# 기존 호환성 유지용
recommendations = st.session_state.recommendation_engine.get_personalized_recommendation(
    mbti, temp, weather, season,
    detected_items=detected_items_data.get("items", []),
    style_analysis=style_analysis_data
)

# 통합 추천 결과를 기존 recommendations에 병합
recommendations["outfit_versions"] = unified_recommendations["outfit_versions"]
```

### 2.3 추천 코디 표시 부분

```python
# 통합 추천 결과 사용
outfit_versions = recommendations.get("outfit_versions", [])

if outfit_versions and len(outfit_versions) >= 3:
    # 통합 추천 사용 (성별 + MBTI + 이미지 분석 + 온도/계절)
    for idx, (col, version) in enumerate(zip([col1, col2, col3], outfit_versions[:3])):
        with col:
            st.write(f"**추천 코디 {idx+1}**")
            st.write(f"**{version['style']}**")
            
            st.info(version['description'])
            st.write(f"**아이템:**")
            
            # 아이템 표시
            for item in version['items']:
                st.write(f"• {item}")
            
            # 추천 제품 표시
            st.write("**추천 제품:**")
            for product in version['products']:
                st.write(f"• {product}")
            
            # 가상 피팅/AI 생성용 데이터 저장
            outfit_desc = {
                "items": version['items'],
                "style": version['style'],
                "colors": [item.split()[0] for item in version['items'] if item.split()[0] in ["검은색", "흰색", "빨간색", "파란색", "회색", "베이지", "네이비"]][:2],
                "gender": version['gender']
            }
            current_image_hash = st.session_state.get("last_image_hash", "default")
            cache_key = f"generated_image_{current_image_hash}_{version['style']}_{idx}"
            outfit_data_list.append({
                "col": col,
                "outfit_desc": outfit_desc,
                "style": version['style'],
                "idx": idx,
                "cache_key": cache_key
            })
```

### 2.4 가상 피팅 실행 부분

```python
# 가상 피팅 모드 선택
fitting_mode = st.radio(
    "이미지 생성 방식",
    ["가상 피팅 (추천)", "AI 생성 (실험적)"],
    index=0,
    key="fitting_mode"
)

# 추천 코디 표시
if fitting_mode == "가상 피팅 (추천)":
    for data in outfit_data_list:
        with data["col"]:
            # 캐시 확인
            cache_key = f"virtual_fitting_{data['cache_key']}"
            
            if cache_key not in st.session_state:
                with st.spinner(f"🎨 {data['style']} 스타일 가상 피팅 중..."):
                    # 원본 이미지 사용
                    source_image = user_uploaded_image if user_uploaded_image is not None else image
                    
                    # 가상 피팅 실행
                    fitted_image = st.session_state.virtual_fitting.composite_outfit_on_image(
                        source_image,
                        data["outfit_desc"]["items"],
                        data["outfit_desc"]["gender"]
                    )
                    
                    if fitted_image:
                        st.session_state[cache_key] = fitted_image
                        st.image(fitted_image, caption=f"{data['style']} 스타일 가상 피팅", width='stretch')
                        st.success("✅ 가상 피팅 완료")
                    else:
                        st.warning("⚠️ 가상 피팅 실패 - 의류 영역을 찾을 수 없습니다")
            else:
                # 캐시된 이미지 사용
                cached_image = st.session_state[cache_key]
                st.image(cached_image, caption=f"{data['style']} 스타일 가상 피팅", width='stretch')
                st.success("✅ 가상 피팅 완료 (캐시)")
```

---

## 3. 핵심 메서드 설명

### 3.1 `composite_outfit_on_image()` - 메인 합성 메서드
- **입력**: 원본 이미지, 추천 아이템 리스트, 성별
- **처리 과정**:
  1. YOLO로 의류 영역 탐지 (상의/하의)
  2. Stable Diffusion Inpainting으로 각 영역에 의상 생성
  3. 생성된 이미지를 원본과 블렌딩
- **출력**: 합성된 이미지
- **개선사항**:
  - 리사이즈 최적화: `needs_resize` 플래그로 불필요한 리사이즈 방지
  - 성별 기반 negative prompt: 남성/여성에 맞는 키워드 제거
  - 에러 처리 강화: VAE decode 시그니처 오류, 디바이스 오류 자동 재시도

### 3.2 `detect_clothing_regions()` - 의류 영역 탐지
- YOLO 탐지 결과를 상의/하의/전신으로 분류
- 가장 높은 confidence의 결과만 선택

### 3.3 `_load_inpaint_pipeline()` - 모델 로드
- Stable Diffusion 2 Inpainting 모델 로드
- MPS(GPU) 모드 자동 감지 및 패치 적용
- VAE는 CPU, UNet은 MPS로 배치

### 3.4 `_apply_mps_patches()` - MPS 호환성 패치
- UNet forward 패치: 모든 텐서를 MPS로 이동
- Scheduler step 패치: 텐서 디바이스 일치
- prepare_mask_latents 패치: 마스크를 MPS로 이동 (올바른 시그니처)
- prepare_latents 패치: latent를 MPS로 이동

### 3.5 `_build_inpaint_prompt()` - 프롬프트 생성 (중요 개선)
- **한글 아이템 텍스트를 영어 프롬프트로 변환**
- **색상 정확도 향상**: `EXACTLY {color} color` 명시
- **타입 구체화**:
  - 상의: `long sleeve shirt`, `short sleeve t-shirt`, `shirt`
  - 하의: `slim-fit trousers`, `shorts`, `pants`
- **재질 추출**: `fabric_map`을 통해 면, 린넨, 울 등 재질 정보 포함
- **성별 명시**: `a man wearing...` 또는 `a woman wearing...` 명확히 지정
- **자연스러운 착용감**: `realistic fit, naturally worn, proper draping, natural folds` 키워드 추가
- **고품질 표현**: `high quality photo, professional photography, authentic clothing texture` 추가

### 3.6 `_simple_color_overlay()` - 폴백 메서드
- Inpainting 실패 시 간단한 색상 오버레이 적용
- 바운딩박스 영역에 색상만 변경

### 3.7 `_patch_vae_for_mps()` - VAE 패치
- VAE encode: 입력을 CPU로 이동, 결과를 MPS로 이동
- VAE decode: `generator` 인자 제거 및 CPU 처리

---

## 4. 최신 개선사항

### 4.1 프롬프트 정확도 향상
- **색상 명시**: `EXACTLY {color} color` 추가로 색상 정확도 향상
- **타입 구체화**: 긴팔/반팔, 바지/반바지 구분
- **재질 정보**: 면, 린넨, 울 등 재질 정보 포함
- **성별 명시**: 프롬프트에 `a man` 또는 `a woman` 명시

### 4.2 Negative Prompt 강화
- **성별 기반 제거**: 남성일 경우 여성 관련 키워드 제거, 여성일 경우 남성 관련 키워드 제거
- **예시**:
  - 남성: `"woman, female, women's clothing, women's shoes, high heels, breasts, cleavage, feminine curves"`
  - 여성: `"man, male, men's clothing, men's shoes"`

### 4.3 리사이즈 최적화
- **불필요한 리사이즈 방지**: `needs_resize` 플래그로 원본 크기가 작으면 리사이즈 생략
- **한 번만 리사이즈**: 리사이즈된 경우에만 원본 크기로 복원하여 이중 리사이즈 방지

### 4.4 통합 추천 로직 연동
- **성별 + MBTI + 이미지 분석 + 온도/계절**: 모든 정보를 통합하여 스타일 생성
- **스타일 → 아이템 → 제품**: 일관된 순서로 추천 생성
- **3가지 버전 통일**: 모든 추천 코디가 동일한 형식으로 표시

### 4.5 에러 처리 강화
- **VAE decode 시그니처 오류**: 자동 감지 및 재패치
- **디바이스 오류**: MPS 패치 재적용 및 재시도
- **폴백 메커니즘**: 에러 발생 시 색상 오버레이로 자동 전환

---

## 📦 필요한 라이브러리

```python
pip install opencv-python pillow numpy torch diffusers accelerate
```

---

## 🔧 주요 기술 스택

1. **YOLOv5**: 의류 영역 탐지
2. **Stable Diffusion 2 Inpainting**: 의상 생성
3. **MPS (Metal Performance Shaders)**: Apple Silicon GPU 가속
4. **OpenCV**: 이미지 처리 및 블렌딩
5. **PIL/Pillow**: 이미지 조작

---

## 📝 참고사항

- MPS 모드에서 VAE는 CPU에서 실행 (안정성)
- UNet은 MPS에서 실행 (속도)
- 이미지는 512px로 리사이즈 후 처리 (속도 향상, 불필요 시 생략)
- 마스크 기반 블렌딩으로 자연스러운 합성
- 에러 발생 시 자동으로 색상 오버레이로 폴백
- 프롬프트에 `EXACTLY {color} color` 명시로 색상 정확도 향상
- 성별 기반 negative prompt로 성별 불일치 방지
- 타입 구체화 (long sleeve shirt, slim-fit trousers)로 의류 정확도 향상

---

## 🎯 최신 프롬프트 예시

### 상의 예시
```
a man wearing a black long sleeve shirt, EXACTLY black color, cotton fabric, 
realistic fit, naturally worn, proper draping, natural folds, 
realistic lighting, natural shadows, high quality photo, 
professional photography, authentic clothing texture
```

### 하의 예시
```
a man wearing gray slim-fit trousers, EXACTLY gray color, cotton fabric, 
realistic fit, naturally worn, proper draping, natural folds, 
realistic lighting, natural shadows, high quality photo, 
professional photography, authentic clothing texture
```

### Negative Prompt (남성)
```
woman, female, women's clothing, women's shoes, high heels, 
breasts, cleavage, feminine curves, 
wrong color, mismatched clothes, double clothing, overlay, blur, 
distorted body, unrealistic fabric, old outfit, wrong gender clothing, 
face, head, portrait, drawing, painting, illustration, cartoon, 
anime, unrealistic, fake, artificial, CGI, 3D render, computer graphics
```
