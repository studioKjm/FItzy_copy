"""
외모 및 패션 점수 매기기 시스템
얼굴, 체형, 패션 스타일 등 다양한 요소를 점수화
"""

import numpy as np
from PIL import Image

class ScoringSystem:
    """외모 및 패션 점수 평가 시스템"""
    
    def __init__(self):
        pass
    
    def score_appearance(self, face_info: dict, body_info: dict, image: Image.Image = None) -> dict:
        """외모 점수 평가"""
        scores = {
            "얼굴": 0,
            "체형": 0,
            "전체 외모": 0
        }
        
        # 얼굴 점수 (0-100) - 개선된 로직 (DeepFace + 황금 비율 고려)
        if face_info and face_info.get("detected"):
            face_shape = face_info.get("face_shape", "")
            face_ratio = face_info.get("face_ratio", 1.0)
            
            # DeepFace 결과 활용 (더 정확한 분석)
            age = face_info.get("age", None)
            emotion = face_info.get("emotion", "")
            gender_deepface = face_info.get("gender_deepface", "")
            
            # 얼굴 비율 기반 점수 (황금 비율 0.618 또는 이상적 비율 0.75-0.85 고려)
            # 이상적 얼굴 비율: 0.75-0.85 (약 0.8 부근)
            ideal_ratio = 0.8
            ratio_deviation = abs(face_ratio - ideal_ratio) if face_ratio else 0.3
            
            # 비율 점수 (0-40점): 이상적 비율에 가까울수록 높은 점수
            if ratio_deviation <= 0.05:  # 0.75-0.85 (이상적)
                ratio_score = 40
            elif ratio_deviation <= 0.10:  # 0.70-0.90 (양호)
                ratio_score = 35
            elif ratio_deviation <= 0.15:  # 0.65-0.95 (보통)
                ratio_score = 28
            else:  # 너무 벗어남
                ratio_score = 18
            
            # 얼굴 형태 보정 점수 (0-25점)
            shape_bonus = 0
            if face_shape == "계란형":
                shape_bonus = 20  # 가장 이상적
            elif face_shape == "사각형":
                shape_bonus = 18  # 각진 형태도 좋음
            elif face_shape == "둥근형":
                shape_bonus = 12
            elif face_shape == "길쭉한형":
                shape_bonus = 8
            else:
                shape_bonus = 5
            
            # 눈 크기 보정 (0-10점)
            eye_bonus = 0
            eye_size = face_info.get("eye_size", "")
            if eye_size == "큰 편":
                eye_bonus = 8
            elif eye_size == "작은 편":
                eye_bonus = 2
            
            # 나이 보정 (0-10점): 젊을수록 보너스
            age_bonus = 0
            if age:
                if 20 <= age <= 30:
                    age_bonus = 8  # 가장 이상적 나이대
                elif 18 <= age < 20 or 30 < age <= 35:
                    age_bonus = 5
                elif 15 <= age < 18 or 35 < age <= 40:
                    age_bonus = 3
                else:
                    age_bonus = 1
            
            # 감정 보정 (0-5점): 긍정적 감정 보너스
            emotion_bonus = 0
            positive_emotions = ["happy", "surprise", "neutral"]
            if emotion in positive_emotions:
                emotion_bonus = 3
            elif emotion:
                emotion_bonus = 1
            
            # 기본 점수
            base_score = 15  # 기본 점수
            
            scores["얼굴"] = base_score + ratio_score + shape_bonus + eye_bonus + age_bonus + emotion_bonus
            
            # 점수 범위 제한 (0-100)
            scores["얼굴"] = max(40, min(100, scores["얼굴"]))
            
            # 디버그 정보 (필요시 주석 해제)
            # print(f"DEBUG 얼굴 점수: base={base_score}, ratio={ratio_score}, shape={shape_bonus}, eye={eye_bonus}, age={age_bonus}, emotion={emotion_bonus}, 총={scores['얼굴']}")
        else:
            scores["얼굴"] = 50  # 기본값
        
        # 체형 점수 (0-100)
        if body_info and body_info.get("detected"):
            body_type = body_info.get("body_type", "")
            body_ratio = body_info.get("body_ratio", 1.0)
            
            # 체형 타입 점수
            if "균형잡힌" in body_type:
                scores["체형"] = 85
            elif "어깨가 넓은" in body_type:
                scores["체형"] = 75
            elif "힙이 넓은" in body_type:
                scores["체형"] = 70
            else:
                scores["체형"] = 65
            
            # 체형 비율 보정 (0.9-1.1 사이면 이상적)
            if body_ratio and 0.9 <= body_ratio <= 1.1:
                scores["체형"] += 5
        else:
            scores["체형"] = 50  # 기본값
        
        # 전체 외모 점수 (평균)
        scores["전체 외모"] = int((scores["얼굴"] + scores["체형"]) / 2)
        
        return scores
    
    def score_fashion(self, detected_items: list, style_analysis: dict, 
                     weather: str, season: str, temperature: float = None,
                     image: Image.Image = None) -> dict:
        """패션 점수 평가"""
        scores = {
            "아이템 구성": 0,
            "스타일 일치도": 0,
            "계절 적합성": 0,
            "날씨 적합성": 0,
            "전체 패션": 0
        }
        
        # 아이템 구성 점수 (0-100)
        if detected_items:
            item_count = len(detected_items)
            # 탐지된 아이템 수에 따라 점수 부여
            if item_count >= 3:
                scores["아이템 구성"] = 85
            elif item_count == 2:
                scores["아이템 구성"] = 70
            elif item_count == 1:
                scores["아이템 구성"] = 55
            else:
                scores["아이템 구성"] = 40
            
            # 신뢰도 보정
            avg_confidence = sum(item.get("confidence", 0) for item in detected_items) / len(detected_items)
            scores["아이템 구성"] += int(avg_confidence * 15)  # 최대 15점 보너스
        else:
            scores["아이템 구성"] = 30  # 아이템이 없으면 낮은 점수
        
        scores["아이템 구성"] = min(100, scores["아이템 구성"])
        
        # 스타일 일치도 점수 (0-100)
        if style_analysis and style_analysis.get("text_matches"):
            matches = style_analysis["text_matches"]
            if matches:
                # 최고 유사도 점수 사용
                max_similarity = max(matches.values())
                scores["스타일 일치도"] = int(max_similarity * 100)
                
                # 여러 스타일이 높은 점수를 받으면 보너스
                high_scores = [v for v in matches.values() if v > 0.3]
                if len(high_scores) >= 3:
                    scores["스타일 일치도"] += 10
                
                scores["스타일 일치도"] = min(100, scores["스타일 일치도"])
        else:
            scores["스타일 일치도"] = 50
        
        # 계절 적합성 점수 (0-100) - 개선: 의상 길이/종류 고려
        seasonal_colors = {
            "봄": ["파스텔", "라이트톤", "핑크", "라벤더", "옐로우"],
            "여름": ["화이트", "브라이트", "아쿠아", "화이트", "화이트"],
            "가을": ["어스톤", "뉴트럴", "터키석", "머스타드", "베이지"],
            "겨울": ["다크톤", "딥컬러", "블랙", "네이비", "그레이"]
        }
        
        # 추운 계절 판별 (온도 기반)
        is_cold = temperature is not None and temperature < 10
        is_very_cold = temperature is not None and temperature < 0
        is_warm = temperature is not None and temperature >= 20
        
        # 탐지된 의상 종류 분석
        has_long_clothes = False
        has_short_clothes = False
        
        if detected_items:
            detected_classes = [item.get("class", "") for item in detected_items if item.get("class")]  # None 제외
            detected_classes_en = [item.get("class_en", "") for item in detected_items if item.get("class_en")]  # None 제외
            all_classes = [c.lower() for c in detected_classes + detected_classes_en if c and isinstance(c, str)]  # 빈 문자열 및 None 제외
            
            # 디버그: 의상 클래스 확인 (필요시 주석 해제)
            # print(f"DEBUG 계절: detected_items 개수={len(detected_items)}, all_classes={all_classes}")
            
            # 긴 옷 / 짧은 옷 구분 (더 정확한 매칭)
            has_long_clothes = any(
                "긴팔" in c or "long sleeve" in c or 
                "trousers" in c or ("바지" in c and "반바지" not in c) or
                "아우터" in c or "outwear" in c 
                for c in all_classes
            )
            has_short_clothes = any(
                "반팔" in c or "short sleeve" in c or 
                "반바지" in c or "shorts" in c or 
                ("드레스" in c or "dress" in c) or
                ("상의" in c and "긴팔" not in c)
                for c in all_classes
            )
        
        # 색상 적합성 점수 (0-40점)
        color_score = 0
        if style_analysis:
            detected_color = style_analysis.get("color", "")
            season_colors = seasonal_colors.get(season, [])
            
            if detected_color:
                if any(season_color.lower() in detected_color.lower() for season_color in season_colors):
                    color_score = 35  # 계절 색상 일치
                elif detected_color in ["검은색", "black", "흰색", "white"]:  # 사계절 적합
                    color_score = 25
                else:
                    color_score = 15  # 다른 색상
            else:
                color_score = 20  # 색상 불명확
        else:
            color_score = 15  # 스타일 분석 없음
        
        # 의상 길이/종류 적합성 점수 (0-60점) - 온도에 따라 엄격하게 평가
        length_score = 0
        if is_very_cold:  # 영하 (< 0도)
            if has_long_clothes and not has_short_clothes:
                length_score = 55  # 긴 옷만 있으면 높은 점수
            elif has_short_clothes and not has_long_clothes:
                length_score = 10   # 짧은 옷만 있으면 매우 낮은 점수 (핵심 수정)
            elif has_long_clothes and has_short_clothes:
                length_score = 25   # 둘 다 있으면 중간
            else:
                length_score = 20   # 불확실
        elif is_cold:  # 0-10도
            if has_long_clothes and not has_short_clothes:
                length_score = 50
            elif has_short_clothes and not has_long_clothes:
                length_score = 15   # 추운 날씨에 짧은 옷은 부적합
            elif has_long_clothes and has_short_clothes:
                length_score = 30   # 혼용
            else:
                length_score = 25
        elif is_warm:  # 20도 이상
            if has_short_clothes:
                length_score = 50  # 더운 날씨에 짧은 옷 적합
            elif has_long_clothes:
                length_score = 30
            else:
                length_score = 35
        else:  # 중간 온도 (10-20도)
            length_score = 35  # 둘 다 적합
        
        # 계절 적합성 = 색상 점수 + 길이 점수
        scores["계절 적합성"] = color_score + length_score
        scores["계절 적합성"] = min(100, max(0, scores["계절 적합성"]))
        
        # 디버그 정보 (필요시 주석 해제)
        # print(f"DEBUG 계절 적합성: 온도={temperature}, is_very_cold={is_very_cold}, has_long={has_long_clothes}, has_short={has_short_clothes}")
        # print(f"DEBUG: color_score={color_score}, length_score={length_score}, 총={scores['계절 적합성']}")
        
        # 날씨 적합성 점수 (0-100) - 온도 고려
        weather_base_scores = {
            "맑음": 80,
            "흐림": 75,
            "비": 70,
            "눈": 65,
            "바람": 75
        }
        weather_base = weather_base_scores.get(weather, 70)
        
        # 온도 보정
        if temperature is not None:
            if temperature < 0:  # 영하
                if weather == "눈":
                    weather_base = 60
                else:
                    weather_base = 65
            elif temperature > 25:  # 여름 날씨
                if weather == "맑음":
                    weather_base = 85
                else:
                    weather_base = 75
        
        scores["날씨 적합성"] = weather_base
        
        # 전체 패션 점수 (가중 평균)
        weights = {
            "아이템 구성": 0.3,
            "스타일 일치도": 0.3,
            "계절 적합성": 0.2,
            "날씨 적합성": 0.2
        }
        
        scores["전체 패션"] = int(
            scores["아이템 구성"] * weights["아이템 구성"] +
            scores["스타일 일치도"] * weights["스타일 일치도"] +
            scores["계절 적합성"] * weights["계절 적합성"] +
            scores["날씨 적합성"] * weights["날씨 적합성"]
        )
        
        return scores
    
    def get_score_label(self, score: int) -> str:
        """점수에 따른 레이블 반환"""
        if score >= 90:
            return "🌟 우수"
        elif score >= 80:
            return "⭐ 좋음"
        elif score >= 70:
            return "👍 보통"
        elif score >= 60:
            return "👌 보통 이하"
        else:
            return "⚠️ 개선 필요"
    
    def get_detailed_feedback(self, appearance_scores: dict, fashion_scores: dict, season: str = "") -> list:
        """상세 피드백 생성"""
        feedback = []
        
        # 외모 피드백
        if appearance_scores["얼굴"] < 70:
            feedback.append("💡 얼굴 형태를 살리는 넥라인을 선택하세요")
        if appearance_scores["체형"] < 70:
            feedback.append("💡 체형을 보완하는 실루엣의 옷을 추천합니다")
        
        # 패션 피드백
        if fashion_scores["아이템 구성"] < 70:
            feedback.append("💡 더 다양한 아이템을 추가하여 코디를 완성하세요")
        if fashion_scores["스타일 일치도"] < 70:
            feedback.append("💡 현재 스타일과 더 어울리는 아이템을 선택해보세요")
        if fashion_scores["계절 적합성"] < 70 and season:
            feedback.append(f"💡 {season}에 어울리는 색상으로 변경을 고려해보세요")
        
        return feedback

