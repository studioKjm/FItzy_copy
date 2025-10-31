"""
시각화 유틸리티: YOLO 탐지 결과 박스 그리기
"""

from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont


def draw_detections(image: Image.Image, items: List[Dict]) -> Image.Image:
    """탐지 박스를 이미지에 그려 반환.
    items: [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
    """
    if image is None or not items:
        return image
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for det in items:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = f"{det.get('class','?')} {det.get('confidence',0):.2f}"
        # 라벨 배경
        tw, th = draw.textlength(label), 12
        draw.rectangle([x1, y1 - th - 2, x1 + tw + 6, y1], fill=(0, 255, 0))
        draw.text((x1 + 3, y1 - th - 1), label, fill=(0, 0, 0))
    return img
