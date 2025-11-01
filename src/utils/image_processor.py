"""
이미지 전처리 유틸리티
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from config import IMAGE_SIZE, MAX_IMAGE_SIZE

def preprocess_image(image, target_size=IMAGE_SIZE):
    """이미지를 모델 입력 형태로 전처리"""
    if isinstance(image, Image.Image):
        # PIL Image를 numpy array로 변환
        img_array = np.array(image)
    elif isinstance(image, np.ndarray):
        img_array = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    return img_array

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    """이미지 크기 조정"""
    if isinstance(image, Image.Image):
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image
