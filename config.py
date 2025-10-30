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

# MBTI별 스타일 매핑
MBTI_STYLES = {
    "ENFP": {"style": "자유롭고 컬러풀한", "colors": ["밝은색", "파스텔"], "mood": "활발한"},
    "ISTJ": {"style": "깔끔하고 단정한", "colors": ["무채색", "네이비"], "mood": "차분한"},
    "ESFP": {"style": "트렌디하고 화려한", "colors": ["강렬한색", "메탈릭"], "mood": "에너지틱한"},
    "INTJ": {"style": "미니멀하고 세련된", "colors": ["모노톤", "블랙"], "mood": "고급스러운"},
    "기타": {"style": "균형잡힌", "colors": ["뉴트럴", "베이지"], "mood": "편안한"}
}

# 계절별 코디 가이드
SEASONAL_GUIDE = {
    "봄": {"colors": ["파스텔", "라이트톤"], "materials": ["린넨", "면"], "mood": "상쾌한"},
    "여름": {"colors": ["화이트", "브라이트"], "materials": ["린넨", "시폰"], "mood": "시원한"},
    "가을": {"colors": ["어스톤", "뉴트럴"], "materials": ["니트", "코듀로이"], "mood": "따뜻한"},
    "겨울": {"colors": ["다크톤", "딥컬러"], "materials": ["울", "가죽"], "mood": "우아한"}
}

# 날씨별 코디 가이드
WEATHER_GUIDE = {
    "맑음": {"accessories": ["선글라스", "캡"], "mood": "밝고 활기찬"},
    "흐림": {"accessories": ["가디건", "스카프"], "mood": "차분하고 우아한"},
    "비": {"accessories": ["우산", "레인부츠"], "mood": "실용적이고 스타일리시한"},
    "눈": {"accessories": ["목도리", "장갑"], "mood": "따뜻하고 포근한"},
    "바람": {"accessories": ["바람막이", "헤어밴드"], "mood": "활동적이고 편안한"}
}

# 앱 설정
APP_TITLE = "Fitzy - AI 패션 코디 추천"
DEBUG = True
