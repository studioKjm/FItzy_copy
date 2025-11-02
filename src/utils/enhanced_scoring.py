"""
향상된 점수 계산 시스템
외모, 패션 점수를 더 정교하게 계산하기 위한 개선된 모듈
"""

import numpy as np
from PIL import Image
import cv2


class EnhancedScoringSystem:
    """향상된 점수 계산 시스템 (정교한 분석)"""
    
    def __init__(self):
        pass
    
    def analyze_image_for_scoring(self, image: Image.Image, detected_items: list):
        """이미지를 분석하여 점수 계산에 필요한 추가 정보 추출"""
        analysis = {
            "dominant_colors": self._extract_dominant_colors(image),
            "color_harmony": 0.0,
            "clothing_lengths": {},
            "style_coherence": 0.0
        }
        
        # 의상 길이 분석
        if detected_items:
            for item in detected_items:
                bbox = item.get("bbox", [])
                if len(bbox) == 4:
                    # bbox 영역의 실제 의상 길이 추정
                    item_class = item.get("class_en", "")
                    if "sleeve" in item_class.lower():
                        # 팔 길이 추정 (bbox 비율 기반)
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        aspect_ratio = width / height if height > 0 else 1.0
                        analysis["clothing_lengths"][item_class] = aspect_ratio
        
        return analysis
    
    def _extract_dominant_colors(self, image: Image.Image, k=5):
        """이미지에서 주요 색상 추출"""
        try:
            # PIL Image를 numpy array로 변환
            img_array = np.array(image.resize((150, 150)))  # 리사이즈로 속도 향상
            
            # RGB로 변환 (RGBA인 경우)
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # K-means 클러스터링으로 주요 색상 추출
            from sklearn.cluster import KMeans
            pixels = img_array.reshape(-1, 3)
            
            # 샘플링 (속도 향상)
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # 각 색상의 비율 계산
            unique, counts = np.unique(labels, return_counts=True)
            color_ratios = counts / len(labels)
            
            # (색상, 비율) 쌍 반환
            dominant_colors = [
                (tuple(colors[i]), float(color_ratios[i]))
                for i in range(len(colors))
            ]
            
            return sorted(dominant_colors, key=lambda x: x[1], reverse=True)
        except Exception:
            return []
    
    def calculate_enhanced_appearance_score(self, face_info: dict, body_info: dict, 
                                           image: Image.Image = None) -> dict:
        """향상된 외모 점수 계산"""
        scores = {
            "얼굴": 50,
            "체형": 50,
            "전체 외모": 50,
            "개선 요소": []
        }
        
        if not face_info or not face_info.get("detected"):
            return scores
        
        # 얼굴 분석 (더 정교한 점수 계산)
        face_shape = face_info.get("face_shape", "")
        face_ratio = face_info.get("face_ratio", 1.0)
        
        # 황금 비율 계산 (더 정교한 방식)
        golden_ratio = 1.618
        ideal_face_ratio = 0.75  # 이상적 얼굴 비율
        
        # 비율 점수 (0-50점)
        ratio_deviation = abs(face_ratio - ideal_face_ratio)
        if ratio_deviation <= 0.05:
            ratio_score = 50
        elif ratio_deviation <= 0.10:
            ratio_score = 42
        elif ratio_deviation <= 0.15:
            ratio_score = 35
        elif ratio_deviation <= 0.20:
            ratio_score = 28
        else:
            ratio_score = 20
        
        # 얼굴 형태 점수 (0-30점)
        shape_scores = {
            "계란형": 28,
            "사각형": 25,
            "둥근형": 20,
            "길쭉한형": 18
        }
        shape_score = shape_scores.get(face_shape, 15)
        
        # 대칭성 점수 추정 (얼굴 비율로 대칭성 추정)
        symmetry_score = 0
        if 0.70 <= face_ratio <= 0.90:  # 대칭적인 얼굴 범위
            symmetry_score = 15
        elif 0.65 <= face_ratio < 0.70 or 0.90 < face_ratio <= 0.95:
            symmetry_score = 10
        else:
            symmetry_score = 5
        
        # 눈 크기 보정
        eye_size = face_info.get("eye_size", "")
        eye_bonus = 8 if eye_size == "큰 편" else (2 if eye_size == "작은 편" else 5)
        
        # DeepFace 정보 활용
        age = face_info.get("age", None)
        age_bonus = 0
        if age:
            if 20 <= age <= 30:
                age_bonus = 10
            elif 18 <= age < 20 or 30 < age <= 35:
                age_bonus = 7
            elif 35 < age <= 40:
                age_bonus = 4
            else:
                age_bonus = 2
        
        # 감정 보정
        emotion = face_info.get("emotion", "")
        emotion_bonus = 5 if emotion in ["happy", "surprise"] else (3 if emotion == "neutral" else 1)
        
        scores["얼굴"] = ratio_score + shape_score + symmetry_score + eye_bonus + age_bonus + emotion_bonus
        scores["얼굴"] = max(40, min(100, scores["얼굴"]))
        
        # 체형 점수 (개선)
        if body_info and body_info.get("detected"):
            body_type = body_info.get("body_type", "")
            body_ratio = body_info.get("body_ratio", 1.0)
            
            # 체형 타입 점수 (더 세밀하게)
            body_type_scores = {
                "균형잡힌 체형": 90,
                "어깨가 넓은 체형": 78,
                "힙이 넓은 체형": 75,
                "일반 체형": 70
            }
            
            base_body_score = 70
            for body_key, body_score in body_type_scores.items():
                if body_key in body_type:
                    base_body_score = body_score
                    break
            
            # 비율 보정 (0.85-1.15가 이상적)
            if 0.85 <= body_ratio <= 1.15:
                base_body_score += 10
            elif 0.80 <= body_ratio < 0.85 or 1.15 < body_ratio <= 1.20:
                base_body_score += 5
            
            scores["체형"] = min(100, base_body_score)
        
        # 전체 외모 점수 (가중 평균)
        scores["전체 외모"] = int((scores["얼굴"] * 0.6 + scores["체형"] * 0.4))
        
        return scores
    
    def calculate_enhanced_fashion_score(self, detected_items: list, style_analysis: dict,
                                       weather: str, season: str, temperature: float = None,
                                       image: Image.Image = None) -> dict:
        """향상된 패션 점수 계산"""
        scores = {
            "아이템 구성": 50,
            "스타일 일치도": 50,
            "계절 적합성": 50,
            "날씨 적합성": 50,
            "전체 패션": 50
        }
        
        # 이미지 분석 결과
        image_analysis = None
        if image:
            image_analysis = self.analyze_image_for_scoring(image, detected_items)
        
        # 아이템 구성 점수 (개선)
        if detected_items:
            item_count = len(detected_items)
            
            # 아이템 종류 다양성 점수
            item_types = set()
            for item in detected_items:
                class_en = item.get("class_en", "")
                if "top" in class_en or "상의" in item.get("class", ""):
                    item_types.add("top")
                elif "dress" in class_en or "드레스" in item.get("class", ""):
                    item_types.add("dress")
                elif "trousers" in class_en or "바지" in item.get("class", ""):
                    item_types.add("bottom")
                elif "shorts" in class_en or "반바지" in item.get("class", ""):
                    item_types.add("bottom")
            
            # 기본 점수
            if item_count >= 3:
                base_score = 80
            elif item_count == 2:
                base_score = 65
            elif item_count == 1:
                base_score = 50
            else:
                base_score = 35
            
            # 다양성 보너스
            diversity_bonus = len(item_types) * 5
            scores["아이템 구성"] = min(100, base_score + diversity_bonus)
            
            # 신뢰도 보정
            avg_confidence = sum(item.get("confidence", 0) for item in detected_items) / len(detected_items)
            scores["아이템 구성"] = int(scores["아이템 구성"] * 0.7 + avg_confidence * 100 * 0.3)
        
        # 계절 적합성 (대폭 개선)
        seasonal_score = self._calculate_seasonal_score(
            detected_items, style_analysis, season, temperature, image_analysis
        )
        scores["계절 적합성"] = seasonal_score
        
        # 날씨 적합성 (개선)
        weather_score = self._calculate_weather_score(
            detected_items, style_analysis, weather, temperature
        )
        scores["날씨 적합성"] = weather_score
        
        # 스타일 일치도 (개선)
        if style_analysis and style_analysis.get("text_matches"):
            matches = style_analysis["text_matches"]
            if matches:
                # 최고 유사도 + 평균 유사도 조합
                max_sim = max(matches.values())
                avg_sim = sum(matches.values()) / len(matches)
                scores["스타일 일치도"] = int((max_sim * 0.6 + avg_sim * 0.4) * 100)
                
                # 높은 점수가 많으면 보너스
                high_scores = [v for v in matches.values() if v > 0.3]
                if len(high_scores) >= 3:
                    scores["스타일 일치도"] = min(100, scores["스타일 일치도"] + 10)
        
        # 전체 패션 점수 (가중 평균)
        scores["전체 패션"] = int(
            scores["아이템 구성"] * 0.25 +
            scores["스타일 일치도"] * 0.25 +
            scores["계절 적합성"] * 0.30 +
            scores["날씨 적합성"] * 0.20
        )
        
        return scores
    
    def _calculate_seasonal_score(self, detected_items: list, style_analysis: dict,
                                season: str, temperature: float = None,
                                image_analysis: dict = None) -> int:
        """계절 적합성 점수 계산 (대폭 개선)"""
        if not detected_items:
            return 50
        
        score = 0
        max_score = 100
        
        # 1. 색상 적합성 (30점)
        color_score = 0
        seasonal_colors = {
            "봄": ["파스텔", "핑크", "라벤더", "옐로우", "라이트톤"],
            "여름": ["화이트", "브라이트", "아쿠아", "파스텔"],
            "가을": ["어스톤", "머스타드", "터키석", "베이지", "브라운"],
            "겨울": ["다크톤", "블랙", "네이비", "그레이", "딥컬러"]
        }
        
        if style_analysis:
            detected_color = style_analysis.get("color", "")
            season_colors = seasonal_colors.get(season, [])
            
            if detected_color:
                if any(season_color.lower() in detected_color.lower() for season_color in season_colors):
                    color_score = 28
                elif detected_color in ["검은색", "black", "흰색", "white"]:
                    color_score = 20  # 사계절 적합
                else:
                    color_score = 12
            else:
                color_score = 15
        else:
            color_score = 12
        
        # 2. 의상 길이/종류 적합성 (50점) - 매우 엄격하게
        length_score = 0
        is_very_cold = temperature is not None and temperature < 0
        is_cold = temperature is not None and temperature < 10
        is_warm = temperature is not None and temperature >= 20
        
        # 실제 탐지된 의상 종류 분석
        all_classes = []
        for item in detected_items:
            class_ko = item.get("class", "")
            class_en = item.get("class_en", "")
            if class_ko:
                all_classes.append(class_ko.lower())
            if class_en:
                all_classes.append(class_en.lower())
        
        has_long_sleeve = any(
            "긴팔" in c or "long sleeve" in c 
            for c in all_classes
        )
        has_short_sleeve = any(
            "반팔" in c or "short sleeve" in c 
            for c in all_classes
        )
        has_long_pants = any(
            "바지" in c or "trousers" in c
            for c in all_classes if "반바지" not in c and "shorts" not in c
        )
        has_short_bottom = any(
            "반바지" in c or "shorts" in c
            for c in all_classes
        )
        
        # 온도별 엄격한 평가
        if is_very_cold:  # 영하
            if has_long_sleeve and has_long_pants and not has_short_sleeve and not has_short_bottom:
                length_score = 48  # 완벽
            elif has_long_sleeve and not has_short_sleeve:
                length_score = 35  # 부분 적합
            elif has_short_sleeve or has_short_bottom:
                length_score = 8   # 매우 부적합
            else:
                length_score = 20  # 불확실
        elif is_cold:  # 0-10도
            if has_long_sleeve and not has_short_sleeve:
                length_score = 40
            elif has_short_sleeve and not has_long_sleeve:
                length_score = 15
            else:
                length_score = 25
        elif is_warm:  # 20도 이상
            if has_short_sleeve or has_short_bottom:
                length_score = 45
            elif has_long_sleeve:
                length_score = 30
            else:
                length_score = 35
        else:  # 중간 온도
            length_score = 35
        
        # 3. 재질/패턴 적합성 (20점) - 향후 확장 가능
        material_score = 10  # 기본값 (재질 분석 미구현)
        
        score = color_score + length_score + material_score
        return min(100, max(0, score))
    
    def _calculate_weather_score(self, detected_items: list, style_analysis: dict,
                                weather: str, temperature: float = None) -> int:
        """날씨 적합성 점수 계산"""
        base_scores = {
            "맑음": 85,
            "흐림": 75,
            "비": 70,
            "눈": 65
        }
        
        base_score = base_scores.get(weather, 70)
        
        # 온도 보정
        if temperature is not None:
            if temperature < 0:
                base_score -= 15  # 너무 추움
            elif temperature > 30:
                base_score -= 10  # 너무 더움
            elif 15 <= temperature <= 25:
                base_score += 10  # 이상적 온도
        
        return min(100, max(0, base_score))

