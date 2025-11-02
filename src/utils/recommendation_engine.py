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
    
    def get_personalized_recommendation(self, mbti, temperature, weather, season,
                                       detected_items=None, style_analysis=None):
        """개인화된 코디 추천 (이미지 분석 결과 포함)"""
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
        
        # 이미지 분석 결과 통합 (새로 추가)
        image_based_suggestions = self._integrate_image_analysis(
            detected_items, style_analysis, seasonal_info, weather_info
        )
        
        return {
            "mbti_style": mbti_style,
            "seasonal_info": seasonal_info,
            "weather_info": weather_info,
            "temperature_guidance": temp_guidance,
            "image_suggestions": image_based_suggestions,  # 이미지 기반 추천 추가
            "recommendation_reason": self._generate_recommendation_reason(
                mbti_style, seasonal_info, weather_info, temp_guidance,
                image_based_suggestions  # 이미지 분석 결과도 이유에 포함
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
            
            for item in items[:5]:  # 상위 5개만
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
            # 상의
            "반팔 상의": ["긴팔 재킷", "가디건", "베스트"],
            "긴팔 상의": ["가디건", "후드집업", "스카프"],
            "상의": ["재킷", "가디건", "목도리"],
            
            # 하의
            "바지": ["부츠", "스니커즈", "벨트"],
            "반바지": ["스니커즈", "슬리퍼", "삭스"],
            "스커트": ["부츠", "플랫슈즈", "스타킹"],
            
            # 드레스
            "드레스": ["재킷", "부츠", "가방"],
            "반팔 드레스": ["가디건", "스니커즈", "모자"],
            "긴팔 드레스": ["코트", "부츠", "가방"],
        }
        
        # 한국어와 영어 모두 확인
        for key in [item_class, item_class_en]:
            if key in complementary_map:
                return complementary_map[key]
        
        # 부분 매칭
        if "상의" in item_class or "top" in item_class_en.lower():
            return ["재킷", "가디건", "액세서리"]
        elif "하의" in item_class or "trousers" in item_class_en.lower() or "shorts" in item_class_en.lower():
            return ["신발", "벨트", "삭스"]
        elif "드레스" in item_class or "dress" in item_class_en.lower():
            return ["재킷", "신발", "가방"]
        
        return ["액세서리", "신발", "가방"]  # 기본값
    
    def _generate_image_based_combinations(self, detected_items, style_analysis, seasonal_info):
        """이미지 분석 결과 기반 조합 추천 생성"""
        combinations = []
        
        if not detected_items:
            return combinations
        
        items = detected_items if isinstance(detected_items, list) else detected_items.get("items", [])
        if not items:
            return combinations
        
        # 탐지된 아이템에서 상의/하의/드레스 분류
        tops = [item for item in items if "상의" in item.get("class", "") or "top" in item.get("class_en", "").lower()]
        bottoms = [item for item in items if any(kw in item.get("class", "") for kw in ["바지", "반바지", "스커트"]) or 
                   any(kw in item.get("class_en", "").lower() for kw in ["trousers", "shorts", "skirt"])]
        dresses = [item for item in items if "드레스" in item.get("class", "") or "dress" in item.get("class_en", "").lower()]
        
        # CLIP 색상 분석
        detected_color = None
        if style_analysis and style_analysis.get("color"):
            detected_color = style_analysis["color"]
        elif style_analysis and style_analysis.get("text_matches"):
            # 색상 점수가 가장 높은 것 선택
            color_matches = {k: v for k, v in style_analysis["text_matches"].items() 
                           if any(c in k.lower() for c in ["색", "color", "red", "blue", "black", "white"])}
            if color_matches:
                detected_color = max(color_matches.items(), key=lambda x: x[1])[0]
        
        # 조합 1: 상의 + 하의 조합
        if tops and bottoms:
            top_item = tops[0]
            bottom_item = bottoms[0]
            
            # 계절 색상과 조화롭게
            seasonal_color = seasonal_info.get("colors", ["뉴트럴"])[0] if seasonal_info else "뉴트럴"
            recommended_color = detected_color if detected_color else seasonal_color
            
            combinations.append({
                "type": "상하 분리형",
                "items": [
                    f"{recommended_color} {top_item.get('class', '상의')}",
                    f"{seasonal_color} {bottom_item.get('class', '하의')}",
                    "액세서리"
                ],
                "reason": f"현재 코디를 기반으로 {recommended_color} 톤으로 조화롭게 연출"
            })
        
        # 조합 2: 드레스 기반
        if dresses:
            dress_item = dresses[0]
            dress_color = detected_color if detected_color else seasonal_info.get("colors", ["뉴트럴"])[0]
            
            combinations.append({
                "type": "원피스 스타일",
                "items": [
                    f"{dress_item.get('class', '드레스')}",
                    "재킷 또는 가디건",
                    "부츠 또는 스니커즈"
                ],
                "reason": f"탐지된 {dress_item.get('class', '드레스')}를 중심으로 레이어링 코디"
            })
        
        # 조합 3: 단일 아이템 기반 확장
        if (tops and not bottoms) or (bottoms and not tops):
            single_item = tops[0] if tops else bottoms[0]
            item_type = "상의" if tops else "하의"
            
            combinations.append({
                "type": "단일 아이템 확장",
                "items": [
                    f"{single_item.get('class', item_type)}",
                    self._get_complementary_items(single_item.get("class", ""), single_item.get("class_en", ""))[0],
                    self._get_complementary_items(single_item.get("class", ""), single_item.get("class_en", ""))[1]
                ],
                "reason": f"현재 {single_item.get('class', item_type)}에 조화로운 아이템 추가"
            })
        
        return combinations[:3]  # 최대 3개만

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
        pool = catalog.get(style, catalog["캐주얼"]).get(gender, catalog["캐주얼"]["공용"])
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
    
    def _generate_recommendation_reason(self, mbti_style, seasonal_info, weather_info, temp_guidance, image_suggestions=None):
        """추천 이유 생성 (이미지 분석 결과 포함)"""
        reasons = [
            f"• {mbti_style['style']} 스타일로 {mbti_style['mood']}한 분위기 연출",
            f"• {seasonal_info['mood']}한 {seasonal_info['colors'][0]} 컬러 조합",
            f"• {weather_info['mood']}한 {weather_info['accessories'][0]} 액세서리 추천",
            f"• {temp_guidance['mood']}한 {temp_guidance['material']} 소재 활용"
        ]
        
        # 이미지 분석 결과 기반 이유 추가
        if image_suggestions:
            detected_info = image_suggestions.get("detected_items_info", [])
            if detected_info:
                detected_names = [item["item"] for item in detected_info[:2]]
                reasons.append(f"• 현재 코디의 {', '.join(detected_names)}를 고려한 조화로운 추천")
            
            style_matches = image_suggestions.get("style_matches", {})
            if style_matches:
                top_style = max(style_matches.items(), key=lambda x: x[1])[0]
                reasons.append(f"• 이미지 분석 결과 {top_style} 스타일이 가장 높게 나타남")
            
            color_matches = image_suggestions.get("color_matches", {})
            if color_matches:
                top_color = max(color_matches.items(), key=lambda x: x[1])[0]
                reasons.append(f"• 이미지에서 {top_color} 톤이 주로 감지되어 색상 조합에 반영")
        
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
