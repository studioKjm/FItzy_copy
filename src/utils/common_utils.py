"""
공통 유틸리티 함수
색상 변환, 디바이스 설정 등 재사용 가능한 함수들
"""

import torch
from typing import Dict, Optional, Tuple


# 색상 변환 맵 (한글 -> 영어)
COLOR_MAP = {
    "검은색": "black", "검정": "black", "흰색": "white", "하얀색": "white",
    "빨간색": "red", "빨강": "red", "파란색": "blue", "파랑": "blue",
    "노란색": "yellow", "노랑": "yellow", "초록색": "green", "초록": "green",
    "분홍색": "pink", "분홍": "pink", "보라색": "purple", "보라": "purple",
    "회색": "gray", "회색톤": "gray", "갈색": "brown", "베이지": "beige",
    "카키": "khaki", "네이비": "navy", "오렌지": "orange", "파스텔": "pastel",
    "딥컬러": "deep color", "다크톤": "dark tone", "라이트톤": "light tone"
}


def get_device_info() -> Tuple[str, str]:
    """
    디바이스 정보 반환 (MPS 우선, CPU 폴백)
    
    Returns:
        (device, vae_device) 튜플
    """
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "cpu"
    return "cpu", "cpu"


def translate_color_to_english(color_kr: str) -> Optional[str]:
    """
    한글 색상명을 영어로 변환
    
    Args:
        color_kr: 한글 색상명
        
    Returns:
        영어 색상명 또는 None
    """
    return COLOR_MAP.get(color_kr, None)


def extract_color_from_text(text: str) -> Optional[str]:
    """
    텍스트에서 색상 추출 (한글 또는 영어)
    
    Args:
        text: 아이템 설명 텍스트
        
    Returns:
        영어 색상명 또는 None
    """
    text_lower = text.lower()
    
    # 한글 색상명 먼저 확인
    for kr, en in COLOR_MAP.items():
        if kr in text:
            return en
    
    # 영어 색상명 확인
    for en in COLOR_MAP.values():
        if en.lower() in text_lower:
            return en
    
    return None


def extract_color_bgr(text: str) -> Optional[Tuple[int, int, int]]:
    """
    텍스트에서 색상 추출 (BGR 형식)
    
    Args:
        text: 아이템 설명 텍스트
        
    Returns:
        (B, G, R) 튜플 또는 None
    """
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
        if color_name in text:
            return bgr
    
    return None

