"""
얼굴 프로토타입 관리 유틸리티
기본 얼굴 이미지를 미리 생성하고 저장하여, 의상만 변경하는 최적화된 이미지 생성 지원
"""

import os
from PIL import Image
from typing import Optional, Dict
import json


class FacePrototypeManager:
    """얼굴 프로토타입 생성 및 관리"""
    
    def __init__(self, base_dir: str = "data/prototypes"):
        """
        Args:
            base_dir: 프로토타입 저장 디렉토리
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """메타데이터 로드"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """메타데이터 저장"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def get_prototype_path(self, gender: str) -> str:
        """프로토타입 파일 경로 반환"""
        return os.path.join(self.base_dir, f"face_prototype_{gender}.png")
    
    def has_prototype(self, gender: str) -> bool:
        """프로토타입 존재 여부 확인"""
        path = self.get_prototype_path(gender)
        return os.path.exists(path)
    
    def load_prototype(self, gender: str) -> Optional[Image.Image]:
        """프로토타입 로드"""
        if not self.has_prototype(gender):
            return None
        
        try:
            path = self.get_prototype_path(gender)
            return Image.open(path)
        except Exception as e:
            print(f"⚠️ 프로토타입 로드 실패 ({gender}): {e}")
            return None
    
    def save_prototype(self, gender: str, image: Image.Image):
        """프로토타입 저장"""
        try:
            path = self.get_prototype_path(gender)
            image.save(path, "PNG")
            
            # 메타데이터 업데이트
            self.metadata[f"{gender}_prototype"] = {
                "path": path,
                "created_at": str(os.path.getctime(path)),
                "size": image.size
            }
            self._save_metadata()
            print(f"✅ 프로토타입 저장 완료: {path}")
        except Exception as e:
            print(f"⚠️ 프로토타입 저장 실패: {e}")
    
    def generate_prototype(self, gender: str, generator) -> Optional[Image.Image]:
        """프로토타입 생성 (기본 얼굴만 있는 이미지)"""
        print(f"🎨 {gender} 얼굴 프로토타입 생성 중...")
        
        # 기본 얼굴만 있는 프롬프트
        if gender == "남성":
            gender_keyword = "male model, man"
        elif gender == "여성":
            gender_keyword = "female model, woman"
        else:
            gender_keyword = "model"
        
        # 기본 옷 (단색 티셔츠)만 입은 프로토타입 - 목 아래만
        prompt = f"Fashion photography, full body {gender_keyword} wearing plain white t-shirt and black long pants, neck down only, upper body and full body visible, entire outfit visible, legs visible, standing pose, no face visible, head cropped out, focus on clothing, high quality, fashion magazine style, neutral background, studio lighting, 8k"
        
        negative_prompt = "face, head, facial features, eyes, nose, mouth, chin, forehead, cheek, ear, hair, face visible, showing face, portrait, headshot, close-up face, cropped legs, missing legs, cut off at waist, upper body only, shorts, short pants, blurry, watermark, grainy, signature, cut off, draft, low quality, worst quality, jpeg artifacts"
        
        try:
            # 기존 생성 메서드 재사용 (negative_prompt 전달)
            if hasattr(generator, '_generate_with_stable_diffusion_local'):
                # _generate_with_stable_diffusion_local은 negative_prompt를 파라미터로 받음
                image = generator._generate_with_stable_diffusion_local(prompt, negative_prompt=negative_prompt)
            else:
                # fallback: 일반 생성 메서드 사용
                outfit_desc = {
                    "items": ["plain white t-shirt", "black pants"],
                    "style": "캐주얼",
                    "colors": ["white", "black"],
                    "gender": gender
                }
                image = generator.generate_outfit_image(outfit_desc)
            
            if image:
                self.save_prototype(gender, image)
                return image
            else:
                print(f"⚠️ 프로토타입 생성 실패 ({gender})")
                return None
                
        except Exception as e:
            print(f"⚠️ 프로토타입 생성 오류 ({gender}): {e}")
            return None

