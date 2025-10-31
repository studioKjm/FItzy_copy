"""
배경 제거 및 사람 세그멘테이션 유틸리티
"""

import numpy as np
from PIL import Image
import io

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️ rembg 라이브러리가 없습니다. pip install rembg로 설치하세요.")


def remove_background(image: Image.Image) -> Image.Image:
    """배경을 제거한 이미지 반환"""
    if not REMBG_AVAILABLE:
        return image
    
    try:
        # RGB 모드로 변환 (rembg는 RGB 입력 필요)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # PIL Image를 bytes로 변환
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # 배경 제거
        output_bytes = remove(img_bytes.getvalue())
        
        if output_bytes is None:
            print("배경 제거 결과가 None입니다.")
            return image
        
        # 결과를 PIL Image로 변환
        result_image = Image.open(io.BytesIO(output_bytes))
        
        # RGBA 모드 확인 및 변환
        if result_image.mode == 'RGBA':
            # 투명 배경 이미지 생성
            return result_image
        else:
            # RGB로 변환되어 나온 경우 (배경 제거 실패)
            print(f"배경 제거 결과 모드: {result_image.mode}, 원본으로 반환")
            return image
            
    except Exception as e:
        print(f"배경 제거 오류: {e}")
        import traceback
        traceback.print_exc()
        return image


def extract_person_mask(image: Image.Image) -> np.ndarray:
    """사람 영역 마스크 추출 (0=배경, 255=사람)"""
    if not REMBG_AVAILABLE:
        # 간단한 대체 방법: YOLO로 person 탐지 영역 사용
        return None
    
    try:
        removed_bg = remove_background(image)
        # 알파 채널을 마스크로 사용
        if removed_bg.mode == 'RGBA':
            mask = np.array(removed_bg.split()[3])  # 알파 채널
            return mask
        return None
    except Exception as e:
        print(f"마스크 추출 오류: {e}")
        return None


def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """마스크를 이미지에 적용"""
    if mask is None:
        return image
    
    img_array = np.array(image.convert('RGB'))
    mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
    
    masked_image = (img_array * mask_3d).astype(np.uint8)
    return Image.fromarray(masked_image)

