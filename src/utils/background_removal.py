"""
배경 제거 및 사람 세그멘테이션 유틸리티
"""

import numpy as np
from PIL import Image
import io

# rembg 라이브러리 확인 (여러 방법 시도)
REMBG_AVAILABLE = False
new_session = None

try:
    # 방법 1: 직접 import
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    try:
        # 방법 2: rembg 모듈 먼저 import
        import rembg
        from rembg import remove
        REMBG_AVAILABLE = True
        try:
            from rembg import new_session
        except:
            new_session = None
    except ImportError:
        try:
            # 방법 3: __import__ 사용
            rembg_module = __import__('rembg')
            remove = rembg_module.remove
            REMBG_AVAILABLE = True
            try:
                new_session = rembg_module.new_session
            except:
                new_session = None
        except:
            REMBG_AVAILABLE = False
    
# 실제 작동 여부 확인 (callable인지 확인만, 실제 실행은 하지 않음)
if REMBG_AVAILABLE:
    try:
        # remove 함수가 callable한지 확인
        if not callable(remove):
            REMBG_AVAILABLE = False
    except Exception:
        # 확인 중 오류 발생 시 False로 설정
        REMBG_AVAILABLE = False


def remove_background(image: Image.Image, model_name: str = 'u2net') -> Image.Image:
    """배경을 제거한 이미지 반환
    
    Args:
        image: PIL Image 객체
        model_name: rembg 모델 이름 ('u2net', 'u2net_human_seg', 'silueta', 'isnet-general-use' 등)
    
    Returns:
        RGBA 모드 PIL Image (배경 제거 성공 시) 또는 원본 이미지
    """
    if not REMBG_AVAILABLE:
        return image
    
    try:
        # RGB 모드로 변환 (rembg는 RGB 입력 필요)
        if image.mode != 'RGB':
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
        
        # PIL Image를 bytes로 변환
        img_bytes = io.BytesIO()
        rgb_image.save(img_bytes, format='PNG', quality=95)
        img_bytes.seek(0)
        img_data = img_bytes.getvalue()
        
        # 배경 제거 시도
        output_bytes = None
        error_messages = []
        
        # 먼저 기본 remove 함수 시도 (가장 간단하고 안정적)
        try:
            output_bytes = remove(img_data)
            if output_bytes and len(output_bytes) > 0:
                # 성공적으로 결과 반환됨
                pass
            else:
                output_bytes = None
        except Exception as e:
            error_messages.append(f"기본 모델 실패: {str(e)}")
            output_bytes = None
        
        # 기본 방법 실패 시 세션을 사용한 모델 시도
        if (output_bytes is None or len(output_bytes) == 0) and new_session:
            models_to_try = ['u2net_human_seg', 'u2net', 'silueta', 'isnet-general-use']
            
            for model_name in models_to_try:
                try:
                    session = new_session(model_name)
                    output_bytes = remove(img_data, session=session)
                    
                    if output_bytes and len(output_bytes) > 0:
                        break  # 성공 시 루프 종료
                    else:
                        output_bytes = None
                except Exception as e:
                    error_messages.append(f"{model_name} 실패: {str(e)}")
                    continue
        
        # 모든 방법 실패
        if output_bytes is None or len(output_bytes) == 0:
            import warnings
            warnings.warn(f"배경 제거 실패. 에러: {', '.join(error_messages[:2])}")
            return image
        
        if output_bytes is None or len(output_bytes) == 0:
            return image
        
        # 결과를 PIL Image로 변환
        result_image = Image.open(io.BytesIO(output_bytes))
        
        # RGBA 모드 확인
        if result_image.mode == 'RGBA':
            # 투명 배경 이미지 성공
            return result_image
        elif result_image.mode == 'RGB':
            # RGB로 나온 경우, RGBA로 변환 (알파 채널 추가)
            rgba_image = Image.new('RGBA', result_image.size, (255, 255, 255, 0))
            rgba_image.paste(result_image, (0, 0))
            return rgba_image
        else:
            # 기타 모드인 경우 원본 반환
            return image
            
    except Exception as e:
        # 에러 발생 시 원본 반환
        import warnings
        warnings.warn(f"배경 제거 오류: {e}")
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

