"""
코디 추천 엔진 - 핵심 추천 로직
통합 분석: 성별 + MBTI + 이미지 분석 + 온도/계절 → 스타일 → 아이템 → 제품
"""

from config import MBTI_STYLES, SEASONAL_GUIDE, WEATHER_GUIDE

# 공통 색상 리스트
COMMON_COLORS = ["베이지", "검은색", "흰색", "회색", "빨간색", "파란색", "네이비", "카키", "갈색"]

class RecommendationEngine:
    """통합 코디 추천 엔진"""
    
    def __init__(self):
        self.mbti_styles = MBTI_STYLES
        self.seasonal_guide = SEASONAL_GUIDE
        self.weather_guide = WEATHER_GUIDE
    
    def generate_unified_outfit_recommendations(self, gender, mbti, temperature, weather, season,
                                                 detected_items=None, style_analysis=None):
        """
        통합 추천 생성: 성별 + MBTI + 이미지 분석 + 온도/계절 → 스타일 → 아이템 → 제품
        
        Returns:
            {
                "outfit_versions": [
                    {
                        "style": "트렌디 스타일",
                        "description": "통합 분석 기반 스타일 설명",
                        "items": ["검은색 긴팔 셔츠", "회색 바지", ...],
                        "products": ["아크테릭스 캡", "나이키 테크플리스", ...]
                    },
                    ...
                ],
                "recommendation_reason": [...]
            }
        """
        # 1. MBTI 스타일 정보 (모든 MBTI 지원)
        mbti_style = self.mbti_styles.get(mbti, self.mbti_styles["ENFP"])
        
        # 2. 계절/날씨/온도 정보
        seasonal_info = self.seasonal_guide.get(season, self.seasonal_guide["봄"])
        weather_info = self.weather_guide.get(weather, self.weather_guide["맑음"])
        temp_guidance = self._get_temperature_guidance(temperature)
        
        # 3. 이미지 분석 결과 통합
        image_suggestions = self._integrate_image_analysis(
            detected_items, style_analysis, seasonal_info, weather_info
        )
        
        # 4. 통합 스타일 생성 (성별 + MBTI + 이미지 분석 + 온도/계절)
        unified_styles = self._generate_unified_styles(
            gender, mbti_style, seasonal_info, weather_info, temp_guidance, image_suggestions
        )
        
        # 5. 각 스타일에 대한 아이템 생성
        outfit_versions = []
        for style_info in unified_styles[:3]:  # 최대 3개 버전
            items = self._generate_outfit_items(
                style_info, gender, mbti_style, seasonal_info, weather_info, 
                temp_guidance, image_suggestions
            )
            
            # 6. 아이템 기반 제품 추천
            products = self._generate_product_recommendations(
                items, style_info["style"], gender, mbti_style
            )
            
            outfit_versions.append({
                "style": style_info["style"],
                "description": style_info["description"],
                "items": items,
                "products": products,
                "gender": gender
            })
        
        # 7. 추천 이유 생성
        recommendation_reason = self._generate_recommendation_reason(
            mbti_style, seasonal_info, weather_info, temp_guidance, image_suggestions,
            gender=gender, mbti=mbti, season=season
        )
        
        return {
            "outfit_versions": outfit_versions,
            "recommendation_reason": recommendation_reason,
            "mbti_style": mbti_style,
            "seasonal_info": seasonal_info,
            "weather_info": weather_info,
            "temperature_guidance": temp_guidance,
            "image_suggestions": image_suggestions
        }
    
    def _generate_unified_styles(self, gender, mbti_style, seasonal_info, weather_info, 
                                temp_guidance, image_suggestions):
        """통합 스타일 생성 (성별 + MBTI + 이미지 분석 + 온도/계절)"""
        styles = []
        
        # 스타일 1: MBTI 중심 스타일
        style1 = {
            "style": f"{mbti_style['style']} 스타일",
            "description": (
                f"{mbti_style['description']}. "
                f"{seasonal_info['mood']}한 {seasonal_info['colors'][0]} 톤으로 "
                f"{temp_guidance['mood']}한 {temp_guidance['material']} 소재 활용"
            ),
            "priority": "mbti"
        }
        styles.append(style1)
        
        # 스타일 2: 계절/날씨 중심 스타일
        style2 = {
            "style": f"{seasonal_info['mood']}한 {seasonal_info['materials'][0]} 스타일",
            "description": (
                f"{seasonal_info['mood']}한 {seasonal_info['colors'][0]} 톤의 "
                f"{seasonal_info['materials'][0]} 소재 코디. "
                f"{weather_info['mood']}한 {weather_info['accessories'][0]} 액세서리 추천"
            ),
            "priority": "season"
        }
        styles.append(style2)
        
        # 스타일 3: 이미지 분석 기반 스타일 (이미지가 있는 경우)
        if image_suggestions and image_suggestions.get("style_matches"):
            top_style = max(image_suggestions["style_matches"].items(), key=lambda x: x[1])
            style3 = {
                "style": f"{top_style[0]} 스타일",
                "description": (
                    f"현재 코디 분석 결과 {top_style[0]} 스타일이 높게 매칭됩니다. "
                    f"{mbti_style['style']} 요소를 가미한 조화로운 코디"
                ),
                "priority": "image"
            }
            styles.append(style3)
        else:
            # 이미지 분석 없을 경우 날씨 중심
            style3 = {
                "style": f"{weather_info['mood']}한 스타일",
                "description": (
                    f"{weather_info['mood']}한 날씨에 맞춘 코디. "
                    f"{mbti_style['colors'][0]} 톤 추천"
                ),
                "priority": "weather"
            }
            styles.append(style3)
        
        return styles
    
    def _generate_outfit_items(self, style_info, gender, mbti_style, seasonal_info, 
                              weather_info, temp_guidance, image_suggestions):
        """스타일 기반 아이템 생성"""
        items = []
        
        # 온도 기반 기본 아이템 선택
        if temp_guidance["layer"] == "다층":
            # 겨울용: 상의 + 재킷/코트 + 하의
            top_color = self._get_color_from_palette(mbti_style, seasonal_info, "top")
            bottom_color = self._get_color_from_palette(mbti_style, seasonal_info, "bottom")
            
            items.append(f"{top_color} 긴팔 셔츠")
            items.append(f"{top_color} {temp_guidance['material']} 재킷 또는 코트")
            items.append(f"{bottom_color} 바지")
        elif temp_guidance["layer"] == "중간":
            # 가을/봄용: 상의 + 가디건/재킷 + 하의
            top_color = self._get_color_from_palette(mbti_style, seasonal_info, "top")
            bottom_color = self._get_color_from_palette(mbti_style, seasonal_info, "bottom")
            
            items.append(f"{top_color} 긴팔 셔츠")
            items.append(f"{top_color} 가디건 또는 재킷")
            items.append(f"{bottom_color} 바지")
        elif temp_guidance["layer"] == "단일":
            # 여름/가을용: 상의 + 하의
            top_color = self._get_color_from_palette(mbti_style, seasonal_info, "top")
            bottom_color = self._get_color_from_palette(mbti_style, seasonal_info, "bottom")
            
            items.append(f"{top_color} 반팔 티셔츠")
            items.append(f"{bottom_color} 바지")
        else:  # 최소
            # 여름용: 상의 + 하의
            top_color = self._get_color_from_palette(mbti_style, seasonal_info, "top")
            bottom_color = self._get_color_from_palette(mbti_style, seasonal_info, "bottom")
            
            items.append(f"{top_color} 반팔 티셔츠")
            items.append(f"{bottom_color} 반바지 또는 바지")
        
        # 신발 추가
        if temp_guidance["layer"] in ["다층", "중간"]:
            items.append("부츠 또는 스니커즈")
        else:
            items.append("스니커즈 또는 샌들")
        
        # 이미지 분석 결과가 있으면 조화로운 색상 적용
        if image_suggestions and image_suggestions.get("color_matches"):
            top_colors = sorted(image_suggestions["color_matches"].items(), 
                              key=lambda x: x[1], reverse=True)
            if top_colors:
                # 첫 번째 아이템 색상 업데이트
                detected_color = top_colors[0][0]
                if items:
                    # 한글 색상명으로 변환
                    color_kr = self._translate_color_to_korean(detected_color)
                    if color_kr:
                        items[0] = items[0].replace(
                            items[0].split()[0], color_kr
                        )
        
        return items
    
    def _get_color_from_palette(self, mbti_style, seasonal_info, item_type):
        """MBTI와 계절을 고려한 색상 선택"""
        # MBTI 색상 우선
        mbti_colors = mbti_style.get("colors", [])
        seasonal_colors = seasonal_info.get("colors", [])
        
        # MBTI 색상을 한글 색상명으로 변환
        color_map = {
            "밝은색": "화이트", "파스텔": "파스텔", "컬러풀": "빨간색",
            "강렬한색": "빨간색", "메탈릭": "회색", "다크톤": "검은색",
            "스포티컬러": "네이비", "네이비": "네이비",
            "뉴트럴": "베이지", "베이지": "베이지", "소프트톤": "베이지",
            "블랙": "검은색", "모노톤": "검은색",
            "어스톤": "갈색", "자연스러운컬러": "베이지",
            "무채색": "회색", "심플톤": "회색", "그레이": "회색"
        }
        
        # 첫 번째 MBTI 색상 사용
        if mbti_colors:
            mbti_color = mbti_colors[0]
            return color_map.get(mbti_color, mbti_color)
        
        # 계절 색상 사용
        if seasonal_colors:
            seasonal_color = seasonal_colors[0]
            return color_map.get(seasonal_color, seasonal_color)
        
        # 기본값
        return "검은색" if item_type == "top" else "회색"
    
    def _translate_color_to_korean(self, color_text):
        """색상 텍스트를 한글 색상명으로 변환"""
        color_map = {
            "red": "빨간색", "빨간색": "빨간색", "빨강": "빨간색",
            "blue": "파란색", "파란색": "파란색", "파랑": "파란색",
            "black": "검은색", "검은색": "검은색", "검정": "검은색",
            "white": "흰색", "흰색": "흰색", "하얀색": "흰색",
            "gray": "회색", "회색": "회색", "grey": "회색",
            "brown": "갈색", "갈색": "갈색",
            "beige": "베이지", "베이지": "베이지",
            "navy": "네이비", "네이비": "네이비",
            "yellow": "노란색", "노란색": "노란색", "노랑": "노란색",
            "green": "초록색", "초록색": "초록색", "초록": "초록색",
            "pink": "분홍색", "분홍색": "분홍색", "분홍": "분홍색",
            "purple": "보라색", "보라색": "보라색", "보라": "보라색",
            "orange": "오렌지", "오렌지": "오렌지",
            "khaki": "카키", "카키": "카키"
        }
        
        # 소문자로 변환 후 매칭
        color_lower = color_text.lower()
        for en, kr in color_map.items():
            if en.lower() in color_lower or color_lower in en.lower():
                return kr
        
        return color_text  # 변환 실패 시 원본 반환
    
    def _generate_product_recommendations(self, items, style, gender, mbti_style):
        """아이템 기반 제품 추천 - 아이템에 정확히 맞는 제품 추천"""
        enhanced_products = []
        
        # 아이템을 순서대로 확인하여 각 아이템에 맞는 제품 추천
        for item in items:
            item_lower = item.lower()
            
            # 상의 아이템 처리
            if "셔츠" in item or "티" in item or "상의" in item:
                # 색상 추출 (공통 색상 리스트 사용)
                color = None
                common_colors = ["베이지", "검은색", "흰색", "회색", "빨간색", "파란색", "네이비", "카키", "갈색"]
                for c in common_colors:
                    if c in item:
                        color = c
                        break
                
                # 타입 추출
                item_type = "긴팔" if "긴팔" in item else "반팔" if "반팔" in item else "셔츠"
                
                if gender == "남성":
                    if color:
                        enhanced_products.append(f"유니클로 U 크루넥 티셔츠 ({color})")
                else:
                        enhanced_products.append("유니클로 U 크루넥 티셔츠")
                else:
                    if color:
                        enhanced_products.append(f"자라 크롭 티셔츠 ({color})")
                    else:
                        enhanced_products.append("자라 크롭 티셔츠")
        
            # 재킷/코트 아이템 처리
            elif "재킷" in item or "코트" in item or "가디건" in item:
                # 색상 추출 (공통 색상 리스트 사용)
                color = None
                for c in COMMON_COLORS:
                    if c in item:
                        color = c
                        break
                
                if "코트" in item:
            if gender == "남성":
                        if color:
                            enhanced_products.append(f"코스 코트 ({color})")
                        else:
                            enhanced_products.append("코스 코트")
                    else:
                        if color:
                            enhanced_products.append(f"자라 트렌치코트 ({color})")
                        else:
                            enhanced_products.append("자라 트렌치코트")
                elif "재킷" in item:
                    if gender == "남성":
                        if color:
                            enhanced_products.append(f"유니클로 울 블렌드 재킷 ({color})")
                        else:
                            enhanced_products.append("유니클로 울 블렌드 재킷")
                    else:
                        if color:
                            enhanced_products.append(f"H&M 재킷 ({color})")
                        else:
                            enhanced_products.append("H&M 재킷")
                elif "가디건" in item:
                    if gender == "남성":
                        if color:
                            enhanced_products.append(f"유니클로 가디건 ({color})")
                        else:
                            enhanced_products.append("유니클로 가디건")
                    else:
                        if color:
                            enhanced_products.append(f"자라 가디건 ({color})")
                        else:
                            enhanced_products.append("자라 가디건")
            
            # 하의 아이템 처리
            elif "바지" in item or "하의" in item or "진" in item:
                # 색상 추출 (공통 색상 리스트 사용)
                color = None
                for c in COMMON_COLORS:
                    if c in item:
                        color = c
                        break
                
                if gender == "남성":
                    if color:
                        enhanced_products.append(f"리바이스 511 슬림진 ({color})")
                    else:
                enhanced_products.append("리바이스 511 슬림진")
                else:
                    if color:
                        enhanced_products.append(f"H&M 하이웨스트 진 ({color})")
            else:
                enhanced_products.append("H&M 하이웨스트 진")
        
            # 신발 아이템 처리
            elif "부츠" in item or "스니커" in item or "신발" in item:
                if "부츠" in item:
                    if gender == "남성":
                        enhanced_products.append("닥터마틴 1461")
            else:
                        enhanced_products.append("찰스앤키스 앵클부츠")
                elif "스니커" in item:
                    if gender == "남성":
                        enhanced_products.append("컨버스 척테일러")
                    else:
                        enhanced_products.append("아디다스 스탠스미스")
                else:
                    if gender == "남성":
                        enhanced_products.append("컨버스 척테일러")
                    else:
                        enhanced_products.append("아디다스 스탠스미스")
        
        # 아이템 기반 제품이 3개 미만이면 스타일 기반 제품으로 보완
        if len(enhanced_products) < 3:
            style_products = self.recommend_products(style, gender)
        for product in style_products:
            if product not in enhanced_products:
                enhanced_products.append(product)
                    if len(enhanced_products) >= 3:
                        break
        
        return enhanced_products[:3]  # 최대 3개
    
    def generate_unified_outfit_recommendations(self, gender, mbti, temperature, weather, season,
                                                 detected_items=None, style_analysis=None):
        """
        통합 추천 생성: 성별 + MBTI + 이미지 분석 + 온도/계절 → 스타일 → 아이템 → 제품
        
        Returns:
            {
                "outfit_versions": [
                    {
                        "style": "트렌디 스타일",
                        "description": "통합 분석 기반 스타일 설명",
                        "items": ["검은색 긴팔 셔츠", "회색 바지", ...],
                        "products": ["아크테릭스 캡", "나이키 테크플리스", ...],
                        "gender": "남성"
                    },
                    ...
                ],
                "recommendation_reason": [...]
            }
        """
        # 1. MBTI 스타일 정보
        mbti_style = self.mbti_styles.get(mbti, self.mbti_styles["ENFP"])
        
        # 2. 계절/날씨/온도 정보
        seasonal_info = self.seasonal_guide.get(season, self.seasonal_guide["봄"])
        weather_info = self.weather_guide.get(weather, self.weather_guide["맑음"])
        temp_guidance = self._get_temperature_guidance(temperature)
        
        # 3. 이미지 분석 결과 통합
        image_suggestions = self._integrate_image_analysis(
            detected_items, style_analysis, seasonal_info, weather_info
        )
        
        # 4. 통합 스타일 생성
        unified_styles = self._generate_unified_styles(
            gender, mbti_style, seasonal_info, weather_info, temp_guidance, image_suggestions
        )
        
        # 5. 각 스타일에 대한 아이템 생성
        outfit_versions = []
        for style_info in unified_styles[:3]:
            items = self._generate_outfit_items(
                style_info, gender, mbti_style, seasonal_info, weather_info, 
                temp_guidance, image_suggestions
            )
            
            # 6. 아이템 기반 제품 추천
            products = self._generate_product_recommendations(
                items, style_info["style"], gender, mbti_style
            )
            
            outfit_versions.append({
                "style": style_info["style"],
                "description": style_info["description"],
                "items": items,
                "products": products,
                "gender": gender
            })
        
        # 7. 추천 이유 생성
        recommendation_reason = self._generate_recommendation_reason(
            mbti_style, seasonal_info, weather_info, temp_guidance, image_suggestions,
            gender=gender, mbti=mbti, season=season
        )
        
        return {
            "outfit_versions": outfit_versions,
            "recommendation_reason": recommendation_reason,
            "mbti_style": mbti_style,
            "seasonal_info": seasonal_info,
            "weather_info": weather_info,
            "temperature_guidance": temp_guidance,
            "image_suggestions": image_suggestions
        }
    
    def _generate_unified_styles(self, gender, mbti_style, seasonal_info, weather_info, 
                                temp_guidance, image_suggestions):
        """통합 스타일 생성"""
        styles = []
        
        # 스타일 1: MBTI 중심
        style1 = {
            "style": f"{mbti_style['style']} 스타일",
            "description": (
                f"{mbti_style['description']}. "
                f"{seasonal_info['mood']}한 {seasonal_info['colors'][0]} 톤으로 "
                f"{temp_guidance['mood']}한 {temp_guidance['material']} 소재 활용"
            ),
            "priority": "mbti"
        }
        styles.append(style1)
        
        # 스타일 2: 계절/날씨 중심
        style2 = {
            "style": f"{seasonal_info['mood']}한 {seasonal_info['materials'][0]} 스타일",
            "description": (
                f"{seasonal_info['mood']}한 {seasonal_info['colors'][0]} 톤의 "
                f"{seasonal_info['materials'][0]} 소재 코디. "
                f"{weather_info['mood']}한 {weather_info['accessories'][0]} 액세서리 추천"
            ),
            "priority": "season"
        }
        styles.append(style2)
        
        # 스타일 3: 이미지 분석 기반 또는 날씨 중심
        if image_suggestions and image_suggestions.get("style_matches"):
            top_style = max(image_suggestions["style_matches"].items(), key=lambda x: x[1])
            style3 = {
                "style": f"{top_style[0]} 스타일",
                "description": (
                    f"현재 코디 분석 결과 {top_style[0]} 스타일이 높게 매칭됩니다. "
                    f"{mbti_style['style']} 요소를 가미한 조화로운 코디"
                ),
                "priority": "image"
            }
        else:
            style3 = {
                "style": f"{weather_info['mood']}한 스타일",
                "description": (
                    f"{weather_info['mood']}한 날씨에 맞춘 코디. "
                    f"{mbti_style['colors'][0]} 톤 추천"
                ),
                "priority": "weather"
            }
        styles.append(style3)
        
        return styles
    
    def _generate_outfit_items(self, style_info, gender, mbti_style, seasonal_info, 
                              weather_info, temp_guidance, image_suggestions):
        """스타일 기반 아이템 생성 (각 스타일마다 다른 색상/아이템)"""
        items = []
        
        # 스타일별 색상 팔레트 (각 스타일마다 다른 색상 사용)
        style_priority = style_info.get("priority", "mbti")
        
        if style_priority == "mbti":
            # MBTI 스타일: MBTI 색상 우선
            top_color = self._get_color_from_palette(mbti_style, seasonal_info, "top")
            # 하의는 대비되는 색상
            if top_color in ["검은색", "네이비"]:
                bottom_color = "회색" if top_color == "검은색" else "베이지"
            elif top_color in ["흰색", "베이지"]:
                bottom_color = "네이비" if top_color == "흰색" else "회색"
            else:
            bottom_color = self._get_color_from_palette(mbti_style, seasonal_info, "bottom")
        elif style_priority == "season":
            # 계절 스타일: 계절 색상 우선
            seasonal_colors = seasonal_info.get("colors", [])
            if seasonal_colors:
                top_color = seasonal_colors[0] if len(seasonal_colors) > 0 else "검은색"
                bottom_color = seasonal_colors[1] if len(seasonal_colors) > 1 else "회색"
            else:
                top_color = "검은색"
                bottom_color = "회색"
        else:  # image or weather
            # 이미지/날씨 스타일: 계절 색상의 다른 조합
            seasonal_colors = seasonal_info.get("colors", [])
            if len(seasonal_colors) >= 2:
                top_color = seasonal_colors[1] if len(seasonal_colors) > 1 else seasonal_colors[0]
                bottom_color = seasonal_colors[0] if len(seasonal_colors) > 0 else "회색"
            else:
                top_color = "베이지"
                bottom_color = "네이비"
        
        # 색상을 한글 색상명으로 변환 (공통 메서드 사용)
        top_color = self._get_color_from_palette(mbti_style, seasonal_info, "top")
        bottom_color = self._get_color_from_palette(mbti_style, seasonal_info, "bottom")
        
        # 온도 기반 기본 아이템
        if temp_guidance["layer"] == "다층":
            items.append(f"{top_color} 긴팔 셔츠")
            items.append(f"{top_color} {temp_guidance['material']} 재킷 또는 코트")
            items.append(f"{bottom_color} 바지")
        elif temp_guidance["layer"] == "중간":
            items.append(f"{top_color} 긴팔 셔츠")
            items.append(f"{top_color} 가디건 또는 재킷")
            items.append(f"{bottom_color} 바지")
        elif temp_guidance["layer"] == "단일":
            items.append(f"{top_color} 반팔 티셔츠")
            items.append(f"{bottom_color} 바지")
        else:
            items.append(f"{top_color} 반팔 티셔츠")
            items.append(f"{bottom_color} 반바지 또는 바지")
        
        # 신발 추가
        if temp_guidance["layer"] in ["다층", "중간"]:
            items.append("부츠 또는 스니커즈")
        else:
            items.append("스니커즈 또는 샌들")
        
        return items
    
    
    def get_personalized_recommendation(self, mbti, temperature, weather, season,
                                       detected_items=None, style_analysis=None):
        """개인화된 코디 추천 (기존 호환성 유지)"""
        # MBTI 기반 스타일 분석
        mbti_style = self.mbti_styles.get(mbti, self.mbti_styles["ENFP"])
        
        # 계절별 가이드 적용
        seasonal_info = self.seasonal_guide.get(season, self.seasonal_guide["봄"])
        
        # 날씨별 가이드 적용
        weather_info = self.weather_guide.get(weather, self.weather_guide["맑음"])
        
        # 온도별 추가 고려사항
        temp_guidance = self._get_temperature_guidance(temperature)
        
        # 이미지 분석 결과 통합
        image_based_suggestions = self._integrate_image_analysis(
            detected_items, style_analysis, seasonal_info, weather_info
        )
        
        return {
            "mbti_style": mbti_style,
            "seasonal_info": seasonal_info,
            "weather_info": weather_info,
            "temperature_guidance": temp_guidance,
            "image_suggestions": image_based_suggestions,
            "recommendation_reason": self._generate_recommendation_reason(
                mbti_style, seasonal_info, weather_info, temp_guidance,
                image_based_suggestions, gender=None, mbti=mbti, season=season
            )
        }
    
    def _integrate_image_analysis(self, detected_items, style_analysis, 
                                  seasonal_info, weather_info):
        """이미지 분석 결과를 추천에 통합"""
        suggestions = {
            "detected_items_info": [],
            "style_matches": {},
            "color_matches": {},
            "recommendation_based_on_image": []
        }
        
        # 1. 탐지된 아이템 분석
        if detected_items and len(detected_items) > 0:
            items = detected_items if isinstance(detected_items, list) else detected_items.get("items", [])
            
            for item in items[:5]:
                item_class = item.get("class", "")
                item_class_en = item.get("class_en", "")
                confidence = item.get("confidence", 0)
                
                suggestions["detected_items_info"].append({
                    "item": item_class,
                    "confidence": confidence,
                    "complementary_items": self._get_complementary_items(item_class, item_class_en)
                })
        
        # 2. CLIP 스타일 분석 결과 활용
        if style_analysis and style_analysis.get("text_matches"):
            matches = style_analysis["text_matches"]
            
            # 스타일 키워드 필터링
            style_keywords = ["캐주얼", "포멀", "트렌디", "스포츠", "빈티지", "모던", "로맨틱", "시크"]
            style_scores = {k: v for k, v in matches.items() if k in style_keywords}
            suggestions["style_matches"] = style_scores
            
            # 색상 키워드 필터링
            color_keywords = ["빨간색", "파란색", "검은색", "흰색", "회색", "갈색", "베이지",
                            "노란색", "yellow", "보라색", "purple", "오렌지", "orange",
                            "초록색", "green", "분홍색", "pink", "네이비", "navy", "카키", "khaki"]
            color_scores = {k: v for k, v in matches.items() if k in color_keywords}
            suggestions["color_matches"] = color_scores
        
        # 3. 이미지 기반 조합 추천 생성
        suggestions["recommendation_based_on_image"] = self._generate_image_based_combinations(
            detected_items, style_analysis, seasonal_info
        )
        
        return suggestions
    
    def _get_complementary_items(self, item_class, item_class_en):
        """탐지된 아이템과 조화로운 추가 아이템 추천"""
        complementary_map = {
            "상의": ["하의", "신발", "액세서리"],
            "하의": ["상의", "신발", "벨트"],
            "신발": ["상의", "하의", "양말"],
            "재킷": ["상의", "하의", "스카프"],
            "가방": ["상의", "하의", "액세서리"]
        }
        
        # 영어 클래스명도 확인
        if "top" in item_class_en.lower() or "shirt" in item_class_en.lower():
            return ["바지", "스니커즈", "가디건"]
        elif "bottom" in item_class_en.lower() or "pants" in item_class_en.lower():
            return ["셔츠", "부츠", "벨트"]
        elif "shoe" in item_class_en.lower() or "boot" in item_class_en.lower():
            return ["셔츠", "바지", "양말"]
        
        return complementary_map.get(item_class, ["액세서리", "신발"])
    
    def _generate_image_based_combinations(self, detected_items, style_analysis, seasonal_info):
        """이미지 분석 기반 조합 생성"""
        combinations = []
        
        if not detected_items:
            return combinations
        
        items = detected_items if isinstance(detected_items, list) else detected_items.get("items", [])
        if not items:
            return combinations
        
        # 상의/하의 분류
        tops = [item for item in items if "top" in item.get("class_en", "").lower() or "상의" in item.get("class", "")]
        bottoms = [item for item in items if "bottom" in item.get("class_en", "").lower() or "하의" in item.get("class", "")]
        dresses = [item for item in items if "dress" in item.get("class_en", "").lower() or "드레스" in item.get("class", "")]
        
        # 색상 추출 (CLIP 분석 결과 활용)
        detected_color = None
        if style_analysis and style_analysis.get("text_matches"):
            color_matches = {k: v for k, v in style_analysis["text_matches"].items() 
                           if any(c in k.lower() for c in ["red", "blue", "black", "white", "gray", "빨간", "파란", "검은", "흰", "회색"])}
            if color_matches:
                detected_color = max(color_matches.items(), key=lambda x: x[1])[0]
        
        # 조합 1: 상의 + 하의 조합 (구체적 색상 포함)
        if tops and bottoms:
            top_item = tops[0]
            bottom_item = bottoms[0]
            
            top_class = top_item.get("class", "상의")
            top_class_en = top_item.get("class_en", "").lower()
            bottom_class = bottom_item.get("class", "하의")
            bottom_class_en = bottom_item.get("class_en", "").lower()
            
            # 색상 결정
            top_color = detected_color if detected_color else seasonal_info.get("colors", ["뉴트럴"])[0]
            bottom_color = "회색" if top_color in ["검은색", "흰색"] else "검은색"
            
            # 타입 결정
            if "long sleeve" in top_class_en or "긴팔" in top_class:
                top_type = "긴팔 셔츠"
            elif "short sleeve" in top_class_en or "반팔" in top_class:
                top_type = "반팔 티셔츠"
            else:
                top_type = "셔츠"
            
            if "trousers" in bottom_class_en or "pants" in bottom_class_en or "바지" in bottom_class:
                bottom_type = "바지"
            elif "shorts" in bottom_class_en or "반바지" in bottom_class:
                bottom_type = "반바지"
            else:
                bottom_type = "바지"
            
            combinations.append({
                "type": "상의+하의 조합",
                "items": [
                    f"{top_color} {top_type}",
                    f"{bottom_color} {bottom_type}",
                    "부츠 또는 스니커즈"
                ],
                "reason": f"현재 코디를 기반으로 {top_color} 톤으로 조화롭게 연출"
            })
        
        return combinations[:3]
    
    def recommend_products(self, style: str, gender: str):
        """스타일/성별 기반 구체 제품 추천 (간단 카탈로그)"""
        gender = gender or "공용"
        catalog = {
            "캐주얼": {
                "남성": ["유니클로 U 크루넥 티셔츠", "리바이스 511 슬림진", "컨버스 척테일러"],
                "여성": ["자라 크롭 티셔츠", "H&M 하이웨스트 진", "아디다스 스탠스미스"],
                "공용": ["무신사 스탠다드 스웻셔츠", "뉴발란스 530", "나이키 볼캡"]
            },
            "포멀": {
                "남성": ["지오지아 슬림핏 수트", "럭키슈에뜨 화이트 셔츠", "닥터마틴 1461"],
                "여성": ["앤아더스토리즈 테일러드 블레이저", "COS 와이드 슬랙스", "찰스앤키스 펌프스"],
                "공용": ["유니클로 린넨 블렌드 자켓", "COS 레더 로퍼"]
            },
            "트렌디": {
                "남성": ["아크테릭스 캡", "나이키 테크플리스", "살로몬 XT-6"],
                "여성": ["아더에러 카디건", "자크뮈스 미니백", "온러닝 클라우드"],
                "공용": ["노스페이스 눕시", "뉴발란스 9060"]
            }
        }
        
        # 스타일 매핑 (다양한 스타일명 지원)
        style_mapping = {
            "캐주얼": "캐주얼",
            "포멀": "포멀",
            "트렌디": "트렌디",
            "자유롭고 창의적": "캐주얼",
            "도전적": "트렌디",
            "감각적": "트렌디",
            "활동적": "캐주얼",
            "세련되고 따뜻한": "포멀",
            "리더십": "포멀",
            "단정하고 따뜻한": "포멀",
            "실용적": "포멀",
            "감성적": "캐주얼",
            "신비롭고 세련된": "포멀",
            "감각적": "트렌디",
            "실용적": "캐주얼",
            "독창적": "트렌디",
            "전략적": "포멀",
            "정갈하고 따뜻한": "포멀",
            "전통적": "포멀"
        }
        
        mapped_style = style_mapping.get(style, "캐주얼")
        pool = catalog.get(mapped_style, catalog["캐주얼"]).get(gender, catalog["캐주얼"]["공용"])
        return pool[:3]
    
    def evaluate_current_outfit(self, detected_items, style_analysis, weather: str, season: str):
        """현재 코디 평가 점수 및 피드백 생성"""
        score = 50
        feedback = []
        # 아이템 다양성
        classes = {item.get("class") for item in (detected_items or [])}
        if classes:
            score += min(len(classes) * 5, 15)
            feedback.append("아이템 구성이 일정 수준 확보되었습니다.")
        else:
            feedback.append("아이템 탐지 결과가 부족합니다. 더 명확한 사진을 업로드해 주세요.")
        # 스타일 적합도
        matches = style_analysis.get("text_matches", {}) if style_analysis else {}
        top_sim = max(matches.values()) if matches else 0.0
        score += int(top_sim * 20)
        if top_sim > 0.4:
            feedback.append("사진과 스타일 키워드의 일치도가 양호합니다.")
        else:
            feedback.append("스타일 일치도가 낮습니다. 키워드를 바꿔보세요.")
        # 날씨/계절 적합도(간단 규칙)
        if weather in ("맑음", "바람"):
            score += 5
        if season in ("여름", "봄") and style_analysis and style_analysis.get("color") in ("화이트", "파란색", "라이트톤"):
            score += 3
        score = max(0, min(100, score))
        # 레이블
        label = "우수" if score >= 80 else ("보통" if score >= 60 else "개선 필요")
        return {"score": score, "label": label, "feedback": feedback}
    
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
    
    def _generate_recommendation_reason(self, mbti_style, seasonal_info, weather_info, temp_guidance, image_suggestions=None, gender=None, mbti=None, season=None):
        """추천 이유 생성 (MBTI, 계절 등을 연계한 상세 설명)"""
        reasons = []
        
        # MBTI 기반 설명
        if mbti and mbti_style:
            mbti_desc = mbti_style.get('description', '')
            mbti_mood = mbti_style.get('mood', '')
            reasons.append(f"• **{mbti} 유형**의 {mbti_style['style']} 특성에 맞춰 {mbti_mood}한 분위기를 연출합니다. {mbti_desc}")
        
        # 계절 기반 설명
        if season and seasonal_info:
            season_colors = ', '.join(seasonal_info.get('colors', [])[:2]) if seasonal_info.get('colors') else seasonal_info.get('colors', [''])[0]
            season_materials = seasonal_info.get('materials', [])
            season_mood = seasonal_info.get('mood', '')
            material_text = f"{season_materials[0]} 소재" if season_materials else "적절한 소재"
            reasons.append(f"• **{season}** 계절에 어울리는 {season_mood}한 {season_colors} 톤의 {material_text}를 활용하여 계절감을 살렸습니다.")
        
        # 날씨 기반 설명
        if weather_info:
            weather_mood = weather_info.get('mood', '')
            weather_accessories = weather_info.get('accessories', [])
            if weather_accessories:
                reasons.append(f"• **{weather_info.get('weather', '')}** 날씨에 적합한 {weather_mood}한 스타일로 {weather_accessories[0]} 같은 액세서리와 함께 착용하면 더욱 완성도 높은 코디가 됩니다.")
        
        # 온도 기반 설명
        if temp_guidance:
            temp_mood = temp_guidance.get('mood', '')
            temp_material = temp_guidance.get('material', '')
            temp_layer = temp_guidance.get('layer', '')
            layer_desc = {
                "다층": "겹쳐 입기",
                "중간": "적절한 레이어링",
                "단일": "심플한 구성",
                "최소": "미니멀한 스타일"
            }
            reasons.append(f"• 현재 온도에 맞춰 {temp_mood}한 {temp_material} 소재를 {layer_desc.get(temp_layer, '적절히')} 활용하여 실용성과 스타일을 모두 고려했습니다.")
        
        # 이미지 분석 결과 기반 이유 추가
        if image_suggestions:
            detected_info = image_suggestions.get("detected_items_info", [])
            if detected_info:
                detected_names = [item["item"] for item in detected_info[:2]]
                reasons.append(f"• 현재 착용 중인 **{', '.join(detected_names)}**와 조화롭게 어울리도록 색상과 스타일을 조정했습니다.")
            
            # 색상 매칭 정보
            color_matches = image_suggestions.get("color_matches", {})
            if color_matches:
                top_colors = sorted(color_matches.items(), key=lambda x: x[1], reverse=True)[:2]
                if top_colors:
                    color_names = [color[0] for color in top_colors]
                    reasons.append(f"• 이미지에서 감지된 **{', '.join(color_names)}** 톤을 활용하여 자연스러운 색상 조화를 이루었습니다.")
        
        return reasons
    
