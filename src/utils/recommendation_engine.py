"""
코디 추천 엔진 - 핵심 추천 로직
날씨, MBTI, 텍스트 검색 등을 통합한 추천 시스템
"""

from config import MBTI_STYLES, SEASONAL_GUIDE, WEATHER_GUIDE

class RecommendationEngine:
    """통합 코디 추천 엔진"""
    
    def __init__(self):
        self.mbti_styles = MBTI_STYLES
        self.seasonal_guide = SEASONAL_GUIDE
        self.weather_guide = WEATHER_GUIDE
    
    def get_personalized_recommendation(self, mbti, temperature, weather, season):
        """개인화된 코디 추천"""
        # MBTI 기반 스타일 분석 (기타인 경우 ENFP 기본값 사용)
        if mbti == "기타":
            mbti = "ENFP"
        mbti_style = self.mbti_styles.get(mbti, self.mbti_styles["ENFP"])
        
        # 계절별 가이드 적용
        seasonal_info = self.seasonal_guide.get(season, self.seasonal_guide["봄"])
        
        # 날씨별 가이드 적용
        weather_info = self.weather_guide.get(weather, self.weather_guide["맑음"])
        
        # 온도별 추가 고려사항
        temp_guidance = self._get_temperature_guidance(temperature)
        
        return {
            "mbti_style": mbti_style,
            "seasonal_info": seasonal_info,
            "weather_info": weather_info,
            "temperature_guidance": temp_guidance,
            "recommendation_reason": self._generate_recommendation_reason(
                mbti_style, seasonal_info, weather_info, temp_guidance
            )
        }
    
    def _get_temperature_guidance(self, temperature):
        """온도별 코디 가이드"""
        if temperature < 5:
            return {"layer": "다층", "material": "울", "mood": "따뜻하고 포근한"}
        elif temperature < 15:
            return {"layer": "중간", "material": "니트", "mood": "적당히 따뜻한"}
        elif temperature < 25:
            return {"layer": "단일", "material": "면", "mood": "시원하고 편안한"}
        else:
            return {"layer": "최소", "material": "린넨", "mood": "시원하고 가벼운"}
    
    def _generate_recommendation_reason(self, mbti_style, seasonal_info, weather_info, temp_guidance):
        """추천 이유 생성"""
        reasons = [
            f"• {mbti_style['style']} 스타일로 {mbti_style['mood']}한 분위기 연출",
            f"• {seasonal_info['mood']}한 {seasonal_info['colors'][0]} 컬러 조합",
            f"• {weather_info['mood']}한 {weather_info['accessories'][0]} 액세서리 추천",
            f"• {temp_guidance['mood']}한 {temp_guidance['material']} 소재 활용"
        ]
        return reasons
    
    def search_text_based_outfits(self, query):
        """텍스트 기반 코디 검색"""
        # 주의: CLIP 모델 매칭은 TextBasedSearcher 클래스에서 처리됨
        outfit_categories = {
            "파티용": {
                "items": ["화려한 드레스", "시퀸 원피스", "스팽글 액세서리"],
                "colors": ["골드", "실버", "레드"],
                "mood": "화려하고 우아한"
            },
            "출근룩": {
                "items": ["정장", "블라우스", "슬랙스", "로퍼"],
                "colors": ["네이비", "블랙", "화이트"],
                "mood": "깔끔하고 전문적인"
            },
            "데이트룩": {
                "items": ["로맨틱 원피스", "부드러운 컬러", "우아한 액세서리"],
                "colors": ["핑크", "라벤더", "크림"],
                "mood": "로맨틱하고 우아한"
            }
        }
        
        # 쿼리에서 카테고리 매칭
        for category, info in outfit_categories.items():
            if category in query:
                return {
                    "category": category,
                    "items": info["items"],
                    "colors": info["colors"],
                    "mood": info["mood"]
                }
        
        return {"category": "일반", "items": ["캐주얼 웨어"], "colors": ["뉴트럴"], "mood": "편안한"}
    
    def get_celebrity_style_reference(self, outfit_style):
        """연예인 스타일 참고 제공"""
        # TODO: 실제 연예인 데이터베이스 연동
        celebrity_styles = {
            "캐주얼": "아이유 - 심플하고 깔끔한 스타일",
            "포멀": "김태리 - 세련되고 우아한 스타일",
            "트렌디": "블랙핑크 - 화려하고 개성있는 스타일"
        }
        return celebrity_styles.get(outfit_style, "추천 스타일 참고")
    
    def get_makeup_suggestions(self, outfit_style, mbti):
        """코디에 맞는 화장법 제안"""
        # TODO: 화장법 데이터베이스 연동
        makeup_guide = {
            "캐주얼": "자연스러운 베이스 + 립글로스",
            "포멀": "완벽한 베이스 + 립스틱",
            "트렌디": "아이섀도 + 하이라이터"
        }
        return makeup_guide.get(outfit_style, "자연스러운 메이크업")
