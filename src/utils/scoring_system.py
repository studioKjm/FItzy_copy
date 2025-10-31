"""
외모 및 패션 점수 매기기 시스템
얼굴, 체형, 패션 스타일 등 다양한 요소를 점수화
"""

import numpy as np


class ScoringSystem:
    """외모 및 패션 점수 평가 시스템"""
    
    def __init__(self):
        pass
    
    def score_appearance(self, face_info: dict, body_info: dict) -> dict:
        """외모 점수 평가"""
        scores = {
            "얼굴": 0,
            "체형": 0,
            "전체 외모": 0
        }
        
        # 얼굴 점수 (0-100)
        if face_info and face_info.get("detected"):
            face_shape = face_info.get("face_shape", "")
            face_ratio = face_info.get("face_ratio", 1.0)
            
            # 얼굴 형태 점수 (균형잡힌 형태일수록 높은 점수)
            if face_shape == "계란형":
                scores["얼굴"] = 85
            elif face_shape == "둥근형":
                scores["얼굴"] = 75
            elif face_shape == "길쭉한형":
                scores["얼굴"] = 70
            else:
                scores["얼굴"] = 65
            
            # 얼굴 비율 보정 (0.7-0.9 사이면 이상적)
            if 0.7 <= face_ratio <= 0.9:
                scores["얼굴"] += 5
            
            # 눈 크기 보정
            eye_size = face_info.get("eye_size", "")
            if eye_size == "큰 편":
                scores["얼굴"] += 5
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
                     weather: str, season: str) -> dict:
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
        
        # 계절 적합성 점수 (0-100)
        seasonal_colors = {
            "봄": ["파스텔", "라이트톤", "핑크", "라벤더", "옐로우"],
            "여름": ["화이트", "브라이트", "아쿠아", "화이트", "화이트"],
            "가을": ["어스톤", "뉴트럴", "터키석", "머스타드", "베이지"],
            "겨울": ["다크톤", "딥컬러", "블랙", "네이비", "그레이"]
        }
        
        if style_analysis:
            detected_color = style_analysis.get("color", "")
            season_colors = seasonal_colors.get(season, [])
            
            # 계절 색상과 일치하는지 확인
            if detected_color:
                if any(season_color.lower() in detected_color.lower() for season_color in season_colors):
                    scores["계절 적합성"] = 85
                elif detected_color in ["검은색", "black", "흰색", "white"]:  # 사계절 적합
                    scores["계절 적합성"] = 70
                else:
                    scores["계절 적합성"] = 55
            else:
                scores["계절 적합성"] = 60
        else:
            scores["계절 적합성"] = 60
        
        # 날씨 적합성 점수 (0-100)
        weather_scores = {
            "맑음": 80,
            "흐림": 75,
            "비": 70,
            "눈": 65,
            "바람": 75
        }
        scores["날씨 적합성"] = weather_scores.get(weather, 70)
        
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

