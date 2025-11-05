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

# MBTI별 스타일 매핑 (모든 16가지 타입)
MBTI_STYLES = {
    "ENFP": {
        "style": "자유롭고 창의적, 톡톡 튀는",
        "colors": ["밝은색", "파스텔", "컬러풀"],
        "mood": "활발한",
        "prompt": "Create a fashion coordination that feels free-spirited, colorful, and full of personality — like an artistic ENFP who loves expressing individuality. Include playful mix of colors, patterns, and accessories.",
        "description": "감각적이고 컬러풀한 스트리트 캐주얼, 독창적인 패턴 믹스"
    },
    "ENTP": {
        "style": "도전적, 트렌디, 실험적",
        "colors": ["강렬한색", "메탈릭", "다크톤"],
        "mood": "에너지틱한",
        "prompt": "A trendy and slightly rebellious outfit that stands out — modern street style with a twist, suitable for an ENTP who enjoys new trends.",
        "description": "비대칭 디자인, 유니크한 스트릿웨어"
    },
    "ESFP": {
        "style": "감각적, 외향적, 패션리더",
        "colors": ["강렬한색", "브라이트", "메탈릭"],
        "mood": "에너지틱한",
        "prompt": "Fashion-forward and glamorous outfit with bright colors and bold accessories — for an outgoing ESFP who loves being the center of attention.",
        "description": "화려한 포인트 아이템, 세련된 트렌디 캐주얼"
    },
    "ESTP": {
        "style": "활동적, 스포티, 스타일리시",
        "colors": ["다크톤", "스포티컬러", "네이비"],
        "mood": "활동적인",
        "prompt": "Sporty yet stylish look with a confident vibe — perfect for an energetic ESTP. Think sleek sneakers, cropped jackets, and functional streetwear.",
        "description": "스포티한 스트릿룩, 기능성 캐주얼"
    },
    "ENFJ": {
        "style": "세련되고 따뜻한, 리더형",
        "colors": ["뉴트럴", "베이지", "소프트톤"],
        "mood": "따뜻한",
        "prompt": "Elegant and confident outfit that radiates warmth and charisma — for an ENFJ who leads with empathy.",
        "description": "클래식하면서 감성적인 오피스 캐주얼"
    },
    "ENTJ": {
        "style": "리더십, 포멀, 파워풀",
        "colors": ["다크톤", "블랙", "네이비"],
        "mood": "파워풀한",
        "prompt": "Smart and powerful fashion — tailored blazer, neat lines, and confident color palette for a goal-oriented ENTJ.",
        "description": "포멀 슈트, 심플 블랙&네이비 컬러"
    },
    "ESFJ": {
        "style": "단정하고 따뜻한, 사회적",
        "colors": ["베이지", "파스텔", "소프트톤"],
        "mood": "따뜻한",
        "prompt": "Friendly and approachable outfit — neat and coordinated style with soft tones, perfect for an ESFJ who values harmony.",
        "description": "베이지·파스텔 계열 오피스 캐주얼"
    },
    "ESTJ": {
        "style": "실용적, 정돈된, 전통적",
        "colors": ["무채색", "네이비", "다크톤"],
        "mood": "차분한",
        "prompt": "A well-organized, neat, and practical outfit — conservative yet classy, suitable for an ESTJ.",
        "description": "정장 또는 포멀한 비즈니스 캐주얼"
    },
    "INFP": {
        "style": "감성적, 내향적, 빈티지",
        "colors": ["어스톤", "베이지", "파스텔"],
        "mood": "감성적인",
        "prompt": "Soft, dreamy outfit with earthy tones and cozy layers — for an INFP with a poetic and introspective style.",
        "description": "빈티지 감성, 따뜻한 니트·레이어드룩"
    },
    "INFJ": {
        "style": "신비롭고 세련된",
        "colors": ["모노톤", "파스텔", "소프트톤"],
        "mood": "세련된",
        "prompt": "Mystical and elegant look — monochrome or pastel outfit with thoughtful details for an INFJ.",
        "description": "차분한 톤, 감성적인 미니멀룩"
    },
    "ISFP": {
        "style": "감각적, 예술적, 내추럴",
        "colors": ["뉴트럴", "어스톤", "자연스러운컬러"],
        "mood": "자연스러운",
        "prompt": "Natural, effortless, and aesthetic outfit — relaxed chic style for an artistic ISFP.",
        "description": "자연스러운 컬러, 루즈핏 캐주얼"
    },
    "ISTP": {
        "style": "실용적, 기능적, 심플",
        "colors": ["무채색", "다크톤", "네이비"],
        "mood": "실용적인",
        "prompt": "Minimalist and functional outfit with clean lines — for a practical ISTP who values comfort.",
        "description": "심플한 디자인, 유틸리티웨어"
    },
    "INTP": {
        "style": "독창적, 이성적, 미니멀",
        "colors": ["모노톤", "심플톤", "그레이"],
        "mood": "미니멀한",
        "prompt": "Modern intellectual style — minimalist fashion with a tech or academic vibe, suitable for an INTP.",
        "description": "심플톤, 모던하고 미니멀한 룩"
    },
    "INTJ": {
        "style": "전략적, 차분한, 시크",
        "colors": ["모노톤", "블랙", "그레이"],
        "mood": "시크한",
        "prompt": "Smart and sophisticated outfit — dark colors, modern silhouette, for an INTJ who values precision.",
        "description": "시크한 블랙/그레이 계열 미니멀룩"
    },
    "ISFJ": {
        "style": "정갈하고 따뜻한, 클래식",
        "colors": ["베이지", "소프트톤", "파스텔"],
        "mood": "따뜻한",
        "prompt": "Classic, gentle outfit with neat coordination and warm tones — suitable for an ISFJ.",
        "description": "니트, 셔츠, 단정한 코디"
    },
    "ISTJ": {
        "style": "전통적, 깔끔한, 단정한",
        "colors": ["무채색", "네이비", "다크톤"],
        "mood": "차분한",
        "prompt": "Traditional and tidy outfit — formal and structured style that reflects the orderliness of an ISTJ.",
        "description": "셔츠·슬랙스·단색 포멀룩"
    }
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
