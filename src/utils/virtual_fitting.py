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
from diffusers import DPMSolverMultistepScheduler
from .common_utils import get_device_info, extract_color_from_text, extract_color_bgr, COLOR_MAP


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
        
        # 디바이스 설정 (공통 유틸리티 사용)
        self.device, self.vae_device = get_device_info()
        if self.device == "mps":
            print("🍎 MPS (GPU) 사용 가능 - 빠른 이미지 생성")
        else:
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
    
    def expand_bbox(self, bbox: List[int], image_size: Tuple[int, int], padding: float = 0.1) -> List[int]:
        """
        바운딩박스 확장 (여유 공간 추가)
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_size: (width, height)
            padding: 확장 비율 (0.1 = 10%)
        
        Returns:
            확장된 [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        width, height = image_size
        
        w = x2 - x1
        h = y2 - y1
        
        # 패딩 적용
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)
        
        return [x1, y1, x2, y2]
    
    def create_mask_from_bbox(self, image_size: Tuple[int, int], bbox: List[int]) -> np.ndarray:
        """
        바운딩박스로부터 마스크 생성
        
        Returns:
            mask: (height, width) 0 또는 255
        """
        width, height = image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
        
        # 부드러운 엣지 (블렌딩 개선)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def apply_color_correction(self, source: np.ndarray, target: np.ndarray, 
                              mask: np.ndarray) -> np.ndarray:
        """
        색상 보정 - 합성된 영역의 색상/조명을 원본과 일치
        
        Args:
            source: 합성할 이미지 (BGR)
            target: 원본 이미지 (BGR)
            mask: 합성 영역 마스크
        
        Returns:
            보정된 이미지 (BGR)
        """
        result = target.copy()
        
        # 마스크 영역에서 히스토그램 매칭
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 각 채널별 히스토그램 매칭
        for i in range(3):
            # 히스토그램 매칭 (OpenCV 내장 함수 없으므로 간단히 밝기 조정)
            source_mean = np.mean(source[:, :, i][mask > 0])
            target_mean = np.mean(target[:, :, i][mask > 0]) if np.sum(mask > 0) > 0 else source_mean
            
            if source_mean > 0:
                scale = target_mean / source_mean
                source[:, :, i] = np.clip(source[:, :, i] * scale, 0, 255).astype(np.uint8)
        
        # Alpha blending
        mask_normalized = mask.astype(float) / 255.0
        for i in range(3):
            result[:, :, i] = (source[:, :, i] * mask_normalized + 
                              target[:, :, i] * (1 - mask_normalized)).astype(np.uint8)
        
        return result
    
    def _load_inpaint_pipeline(self):
        """Stable Diffusion Inpainting 파이프라인 로드"""
        if self.inpaint_pipe is not None:
            return
        
        print("🎨 Stable Diffusion Inpainting 모델 로드 중...")
        print("   - 의류 영역을 실제로 교체합니다 (색상 오버레이가 아님)")
        print("   - 처음 실행 시 모델 다운로드 (약 5GB, 몇 분 소요)")
        print(f"   - 장치: {self.device.upper()} 모드 ({'빠름' if self.device == 'mps' else '느림'})")
        
        try:
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float32,
                safety_checker=None,
                device_map=None
            )
            
            # PNDM 대신 더 빠르고 안정적인 DPM Solver 스케줄러 사용
            self.inpaint_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.inpaint_pipe.scheduler.config
            )
            
            # 디바이스 배치 (MPS: UNet만, CPU: VAE/TextEncoder)
            if self.device == "mps":
                self.inpaint_pipe.unet = self.inpaint_pipe.unet.float().to(self.device, non_blocking=False)
                self.inpaint_pipe.vae = self.inpaint_pipe.vae.to(self.vae_device, non_blocking=False)
                self.inpaint_pipe.text_encoder = self.inpaint_pipe.text_encoder.float().to("cpu", non_blocking=False)
                
                # MPS 패치 적용 (순서 중요: VAE 패치 먼저)
                self._patch_vae_for_mps()
                self._apply_mps_patches()
                
                print("✅ Inpainting 모델 로드 완료 (MPS/GPU 모드, DPM Solver 스케줄러)")
            else:
                # CPU 모드
                self.inpaint_pipe.unet = self.inpaint_pipe.unet.to("cpu")
                self.inpaint_pipe.vae = self.inpaint_pipe.vae.to("cpu")
                self.inpaint_pipe.text_encoder = self.inpaint_pipe.text_encoder.to("cpu")
                print("✅ Inpainting 모델 로드 완료 (CPU 모드, DPM Solver 스케줄러)")
        except Exception as e:
            print(f"⚠️ Inpainting 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            self.inpaint_pipe = None
    
    def _patch_vae_for_mps(self):
        """VAE의 encode/decode 메서드를 패치하여 MPS와 호환되도록"""
        if self.device != "mps":
            return
        
        # VAE encode 패치
        original_encode = self.inpaint_pipe.vae.encode
        
        def patched_vae_encode(self_vae, x, return_dict=True, **kwargs):
            # VAE는 CPU에 있으므로 입력은 CPU로
            if x.device.type != "cpu":
                x = x.to("cpu", non_blocking=False)
            # VAE encode 실행
            result = original_encode(x, return_dict=return_dict, **kwargs)
            # latents를 MPS로 이동 (필요한 경우)
            if return_dict:
                if hasattr(result, 'latent_dist'):
                    # latent_dist의 sample을 MPS로 이동
                    pass  # sample() 호출 시 처리
                return result
            else:
                # 튜플 반환인 경우
                if isinstance(result, tuple):
                    return tuple(r.to(self.device, non_blocking=False) if isinstance(r, torch.Tensor) and r.device.type != self.device else r for r in result)
                return result.to(self.device, non_blocking=False) if isinstance(result, torch.Tensor) and result.device.type != self.device else result
        
        self.inpaint_pipe.vae.encode = patched_vae_encode.__get__(self.inpaint_pipe.vae, type(self.inpaint_pipe.vae))
        
        # VAE decode 패치 (강화된 버전) - generator 인자 명시적으로 처리
        original_decode = self.inpaint_pipe.vae.decode
        
        # 원본 함수의 시그니처 확인을 위해 inspect 사용
        import inspect
        sig = inspect.signature(original_decode)
        print(f"   📋 VAE decode 원본 시그니처: {sig}")
        
        def patched_vae_decode(self_vae, z, return_dict=True, generator=None, **kwargs):
            # generator 인자를 명시적으로 받되 무시 (에러 방지)
            # 입력을 CPU로 이동
            if z.device.type != "cpu":
                z = z.to("cpu", non_blocking=False)
            # generator와 관련된 모든 인자 제거
            kwargs.pop('generator', None)
            # 원본 decode 호출 (generator 제외)
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
            # 모든 입력을 MPS로 이동
            if sample.device.type != self.device:
                sample = sample.to(self.device, non_blocking=False)
            if isinstance(timestep, torch.Tensor) and timestep.device.type != self.device:
                timestep = timestep.to(self.device, non_blocking=False)
            if encoder_hidden_states is not None and encoder_hidden_states.device.type != self.device:
                encoder_hidden_states = encoder_hidden_states.to(self.device, non_blocking=False)
            
            # kwargs의 텐서도 MPS로
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
        
        # Inpainting 파이프라인의 __call__ 메서드 패치 (가장 중요!)
        original_call = self.inpaint_pipe.__call__
        
        def patched_call(self_pipe, prompt=None, image=None, mask_image=None, **kwargs):
            # 모든 입력 이미지/마스크를 CPU에서 처리 (VAE가 CPU에 있으므로)
            # 하지만 latent 생성 후에는 MPS로 이동해야 함
            
            # 원본 호출
            result = original_call(prompt=prompt, image=image, mask_image=mask_image, **kwargs)
            return result
        
        # Inpainting 파이프라인의 _encode_vae_image 패치 (masked_image_latents를 MPS로 이동)
        if hasattr(self.inpaint_pipe, '_encode_vae_image'):
            original_encode_vae_image = self.inpaint_pipe._encode_vae_image
            
            def patched_encode_vae_image(self_pipe, image, generator):
                # VAE는 CPU에 있으므로 이미지도 CPU로
                if image.device.type != "cpu":
                    image = image.to("cpu", non_blocking=False)
                # VAE encode 실행
                result = original_encode_vae_image(image, generator)
                # 결과를 MPS로 이동
                if isinstance(result, torch.Tensor):
                    if result.device.type != self.device:
                        result = result.to(self.device, non_blocking=False)
                return result
            
            self.inpaint_pipe._encode_vae_image = patched_encode_vae_image.__get__(self.inpaint_pipe, type(self.inpaint_pipe))
        
        # 가장 중요한 패치: prepare_mask_latents와 prepare_latents
        # Inpainting 파이프라인의 내부 메서드를 직접 패치
        import types
        
        # prepare_mask_latents 패치 (올바른 시그니처)
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
        
        # prepare_latents 패치 (사전에 적용, 올바른 시그니처)
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
        
        # __call__ 메서드 패치 불필요 - VAE decode 패치로 충분
        # (이미 _patch_vae_for_mps에서 처리됨)
        
        print("   ✅ MPS 패치 적용 완료")
    
    def composite_outfit_on_image(self, original_image: Image.Image, 
                                 outfit_items: List[str],
                                 gender: str = "남성") -> Optional[Image.Image]:
        """
        원본 이미지에 추천 코디를 합성
        
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
                # 영역이 없어도 원본 이미지에 텍스트로 표시
                result_image = self._create_text_overlay_image(original_image, outfit_items)
                return result_image, []  # 프롬프트 정보 없음
            
            # 2. OpenCV 형식으로 변환
            img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]
            
            # 3. Inpainting으로 실제 의류 합성
            # Inpainting 파이프라인 로드
            self._load_inpaint_pipeline()
            
            if self.inpaint_pipe is None:
                print("⚠️ Inpainting 모델 없음. 간단한 색상 오버레이 사용")
                result_image = self._simple_color_overlay(img_cv, regions, outfit_items, width, height)
                return result_image, []  # 프롬프트 정보 없음
            
            # Inpainting으로 각 아이템 합성 (상의 + 하의 모두 처리)
            result_pil = original_image.copy()
            prompts_info = []  # 프롬프트 정보 저장
            
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
                
                # 프롬프트 정보 저장
                region_name = "상의" if region_type == "top" else "하의"
                prompts_info.append({
                    "region": region_name,
                    "prompt": prompt
                })
                
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
                    # 이미지와 마스크를 최적 크기로 리사이즈 (속도 향상)
                    # 원본 크기에 비례하여 리사이즈 (너무 크면 느림)
                    max_size = 512  # 최대 크기 제한
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
                    
                    # Inpainting 실행 (DPM Solver는 더 적은 스텝으로도 좋은 결과)
                    # 스텝 수 조정: IndexError 방지를 위해 11로 설정 (DPM Solver는 내부적으로 +1을 사용)
                    num_steps = 11 if self.device == "mps" else 7
                    
                    with torch.no_grad():
                        try:
                            # 스케줄러 초기화 (매번 새로 시작)
                            if hasattr(self.inpaint_pipe.scheduler, 'set_timesteps'):
                                self.inpaint_pipe.scheduler.set_timesteps(num_steps, device=self.device)
                            
                            result = self.inpaint_pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                image=result_pil_for_inpaint,
                                mask_image=mask_pil_for_inpaint,
                                num_inference_steps=num_steps,  # 감소된 스텝 수
                                guidance_scale=7.5,  # 적절한 가이던스 (9.0 → 7.5)
                                strength=0.85  # 약간 낮춤 (0.9 → 0.85)
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
                                # 재시도 (스케줄러 초기화)
                                num_steps = 12 if self.device == "mps" else 8
                                if hasattr(self.inpaint_pipe.scheduler, 'set_timesteps'):
                                    self.inpaint_pipe.scheduler.set_timesteps(num_steps, device=self.device)
                                result = self.inpaint_pipe(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    image=result_pil_for_inpaint,
                                    mask_image=mask_pil_for_inpaint,
                                    num_inference_steps=num_steps,
                                    guidance_scale=7.5,
                                    strength=0.85
                                )
                            elif "must be on the same device" in error_str or "same device" in error_str:
                                # 디바이스 오류 - MPS 패치 재적용
                                print(f"   ⚠️ 디바이스 오류, MPS 패치 재적용 중...")
                                # 패치 재적용
                                self._apply_mps_patches()
                                # 재시도 (스케줄러 초기화)
                                num_steps = 12 if self.device == "mps" else 8
                                if hasattr(self.inpaint_pipe.scheduler, 'set_timesteps'):
                                    self.inpaint_pipe.scheduler.set_timesteps(num_steps, device=self.device)
                                result = self.inpaint_pipe(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    image=result_pil_for_inpaint,
                                    mask_image=mask_pil_for_inpaint,
                                    num_inference_steps=num_steps,
                                    guidance_scale=7.5,
                                    strength=0.85
                                )
                            elif "list index out of range" in error_str or "IndexError" in error_str:
                                # 스케줄러 초기화 오류 - 스케줄러 재초기화
                                print(f"   ⚠️ 스케줄러 초기화 오류, 재초기화 중...")
                                # 스케줄러 재생성
                                from diffusers import DPMSolverMultistepScheduler
                                self.inpaint_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                                    self.inpaint_pipe.scheduler.config
                                )
                                # 재시도 (IndexError 방지를 위해 스텝 수 조정)
                                num_steps = 11 if self.device == "mps" else 7
                                if hasattr(self.inpaint_pipe.scheduler, 'set_timesteps'):
                                    self.inpaint_pipe.scheduler.set_timesteps(num_steps, device=self.device)
                                result = self.inpaint_pipe(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    image=result_pil_for_inpaint,
                                    mask_image=mask_pil_for_inpaint,
                                    num_inference_steps=num_steps,
                                    guidance_scale=7.5,
                                    strength=0.85
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
                    
                    mask_np = np.array(mask_pil_full) > 127  # 이진 마스크 (boolean)
                    mask_3d = np.stack([mask_np] * 3, axis=2).astype(float)  # 0.0 또는 1.0
                    
                    # 마스크 영역은 생성된 이미지, 나머지는 원본
                    # mask_3d가 1인 영역 = 생성된 이미지, 0인 영역 = 원본
                    blended = result_np.astype(float) * (1.0 - mask_3d) + generated_np.astype(float) * mask_3d
                    result_np = np.clip(blended, 0, 255).astype(np.uint8)
                    
                    result_pil = Image.fromarray(result_np)
                    
                    print(f"✅ {region_type} 영역 Inpainting 완료 (실제 합성됨)")
                    print(f"   - 마스크 영역 크기: {np.sum(mask_np)} 픽셀")
                    
                except Exception as e:
                    print(f"⚠️ Inpainting 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    # 폴백: 간단한 색상 오버레이
                    result_image = self._simple_color_overlay(img_cv, regions, outfit_items, width, height)
                    return result_image, []  # 프롬프트 정보 없음
            
            print("✅ 가상 피팅 완료 (Inpainting)")
            # 프롬프트 정보와 함께 반환
            return result_pil, prompts_info
            
        except Exception as e:
            print(f"⚠️ 가상 피팅 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, []  # 프롬프트 정보 없음
    
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
        # 색상 변환 (공통 유틸리티 사용)
        
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
        item_text_lower = item_text.lower()
        
        # 색상 추출 (공통 유틸리티 사용)
        extracted_color = extract_color_from_text(item_text)
        if extracted_color:
            # 색상명 제거하여 타입만 남김
            for kr, en in COLOR_MAP.items():
            if kr in item_text:
                    en_item = en_item.replace(kr, "").strip()
                if en.lower() in item_text_lower:
                    en_item = en_item.replace(en, "").strip()
        
        # 의류 타입 추출 (더 정확하게)
        extracted_type = None
        # 긴팔/반팔 먼저 확인
        if "긴팔" in item_text or "long sleeve" in item_text_lower:
            extracted_type = "long sleeve"
        elif "반팔" in item_text or "short sleeve" in item_text_lower:
            extracted_type = "short sleeve"
        
        # 그 다음 셔츠/티셔츠/바지 등 확인
        for kr, en in item_map.items():
            if kr in item_text:
                if extracted_type:
                    # 이미 긴팔/반팔이 있으면 조합
                    if "sleeve" in en:
                        extracted_type = f"{extracted_type} {en.replace('sleeve', '').strip()}"
                    else:
                        extracted_type = f"{extracted_type} {en}"
                else:
                extracted_type = en
                en_item = en_item.replace(kr, "")
        
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
                if "long sleeve" in extracted_type.lower() or "긴팔" in item_text:
                    type_spec = "long sleeve shirt"
                elif "short sleeve" in extracted_type.lower() or "반팔" in item_text:
                    type_spec = "short sleeve t-shirt"
                elif "t-shirt" in extracted_type.lower() or "티" in item_text:
                    type_spec = "t-shirt"
                else:
                    type_spec = "shirt"
                
                # 색상이 정확히 반영되도록 강조
                prompt = (
                    f"a {gender_kw} wearing a {extracted_color} {type_spec}, "
                    f"EXACTLY {extracted_color} color, {fabric_part}, "
                    f"realistic fit, naturally worn, proper draping, natural folds, "
                    f"realistic lighting, natural shadows, high quality photo, "
                    f"professional photography, authentic clothing texture"
                )
                print(f"   📝 프롬프트 생성: 색상={extracted_color}, 타입={type_spec}")
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
                if "pants" in extracted_type.lower() or "바지" in item_text:
                    type_spec = "slim-fit trousers"
                elif "shorts" in extracted_type.lower() or "반바지" in item_text:
                    type_spec = "shorts"
                else:
                    type_spec = "pants"
                
                # 색상이 정확히 반영되도록 강조
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
        """
        폴백: 간단한 색상 오버레이 (Inpainting 실패 시)
        """
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
        """
        아이템 텍스트에서 목표 색상 추출 (BGR)
        
        Returns:
            (B, G, R) 또는 None
        """
        return extract_color_bgr(item_text)
    
    def _create_text_overlay_image(self, image: Image.Image, items: List[str]) -> Image.Image:
        """
        의류 탐지 실패 시 원본 이미지에 텍스트 오버레이
        
        Args:
            image: 원본 이미지
            items: 추천 아이템 리스트
        
        Returns:
            텍스트가 추가된 이미지
        """
        from PIL import ImageDraw
        
        # PIL 이미지 복사
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # 텍스트 추가
        text_lines = ["추천 코디:"] + items
        y_offset = 20
        
        for line in text_lines:
            # 배경 박스
            text_bbox = draw.textbbox((10, y_offset), line)
            draw.rectangle(
                [(text_bbox[0]-5, text_bbox[1]-5), (text_bbox[2]+5, text_bbox[3]+5)], 
                fill=(255, 255, 255)
            )
            # 텍스트
            draw.text((10, y_offset), line, fill=(0, 0, 0))
            y_offset += 25
        
        return img_with_text
