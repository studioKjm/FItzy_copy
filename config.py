"""
Fitzy 앱 설정 파일
모델 경로, 하이퍼파라미터, 데이터셋 경로 등 설정
"""

import os

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 모델 설정
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "weights", "yolov5_fashion.pt")
CLIP_MODEL_NAME = "ViT-B/32"  # CLIP 모델 버전

# 데이터셋 경로
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 이미지 처리 설정
IMAGE_SIZE = (640, 640)  # YOLO 입력 크기
MAX_IMAGE_SIZE = 1024  # 최대 이미지 크기

# 탐지 클래스 (패션 아이템)
FASHION_CLASSES = [
    "상의", "하의", "신발", "모자", "가방", "액세서리"
]

# 스타일 키워드 (CLIP 분석용)
STYLE_KEYWORDS = [
    "캐주얼", "정장", "스포츠", "빈티지", "모던",
    "빨간색", "파란색", "검은색", "흰색", "회색"
]

# 앱 설정
APP_TITLE = "Fitzy - AI 패션 코디 추천"
DEBUG = True
