"""
AI 모델 관련 클래스들
YOLOv5와 CLIP 모델을 활용한 옷 탐지 및 스타일 분석
"""

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from config import YOLO_MODEL_PATH, CLIP_MODEL_NAME, FASHION_CLASSES
import os

class YOLODetector:
    """YOLOv5를 사용한 옷 아이템 탐지 클래스"""
    
    def __init__(self, model_path=None):
        """YOLOv5 모델 초기화"""
        if model_path is None:
            model_path = YOLO_MODEL_PATH
        
        # 모델 파일이 없으면 사전 학습된 모델 사용 (yolov5n, yolov5s 등)
        if not os.path.exists(model_path):
            print(f"모델 파일이 없습니다: {model_path}")
            print("사전 학습된 YOLOv5 모델을 사용합니다: yolov5n")
            # COCO 사전 학습 모델 사용 (person, bag 등 일반 객체 탐지)
            self.model = YOLO('yolov5n.pt')
            print("일반 객체 탐지 모델로 동작합니다. 패션 전용 모델 학습이 필요합니다.")
        else:
            self.model = YOLO(model_path)
            print(f"YOLOv5 모델 로드 완료: {model_path}")
    
    def detect_clothes(self, image):
        """이미지에서 옷 아이템 탐지"""
        # 이미지 전처리
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # YOLOv5 추론
        results = self.model(img_array, verbose=False)
        
        # 결과 파싱
        detected_items = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # COCO 클래스 이름 가져오기
                class_name = self.model.names[class_id]
                
                # 패션 관련 객체만 필터링 (person, bag, backpack 등)
                fashion_related = ['person', 'handbag', 'backpack', 'suitcase', 'sports ball']
                if class_name in fashion_related or confidence > 0.3:
                    detected_items.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })
        
        return {
            "items": detected_items,
            "image_size": image.size if isinstance(image, Image.Image) else (img_array.shape[1], img_array.shape[0])
        }

class CLIPAnalyzer:
    """CLIP 모델을 사용한 스타일 분석 클래스"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """CLIP 모델 초기화"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CLIP 모델 로드 중... (장치: {self.device})")
        
        try:
            # 모델을 먼저 CPU에 로드한 후 device로 이동
            # device_map="cpu"를 사용하여 메타 텐서 문제 방지
            self.model = CLIPModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None  # 먼저 CPU에 로드
            )
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # 모델이 완전히 로드된 후 device로 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            else:
                # CPU인 경우 이미 CPU에 있으므로 이동 불필요
                pass
                
            self.model.eval()
            print(f"CLIP 모델 로드 완료: {model_name} (장치: {self.device})")
        except Exception as e:
            print(f"CLIP 모델 로드 실패: {e}")
            print("첫 실행 시 인터넷 연결이 필요합니다.")
            # 대체 방법 시도
            try:
                print("대체 방법으로 모델 로드 시도...")
                self.model = CLIPModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                self.processor = CLIPProcessor.from_pretrained(model_name)
                # device 이동 없이 CPU에서 사용
                self.device = "cpu"
                self.model.eval()
                print(f"CLIP 모델 로드 완료 (CPU 모드): {model_name}")
            except Exception as e2:
                print(f"대체 방법도 실패: {e2}")
                raise
    
    def analyze_style(self, image, text_descriptions):
        """이미지의 스타일과 색상 분석"""
        if not text_descriptions:
            text_descriptions = ["캐주얼", "포멀", "트렌디", "빨간색", "파란색", "검은색", "흰색"]
        
        # 이미지 전처리
        if isinstance(image, Image.Image):
            pass  # PIL Image는 그대로 사용
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        try:
            # 이미지와 텍스트 처리
            inputs = self.processor(
                text=text_descriptions,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 이미지-텍스트 유사도 계산
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # 정규화
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 코사인 유사도 (스케일링)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 결과 파싱
            similarities = similarity[0].cpu().numpy()
            text_matches = {
                desc: float(sim) for desc, sim in zip(text_descriptions, similarities)
            }
            
            # 가장 유사한 스타일 찾기
            best_match_idx = similarities.argmax()
            best_style = text_descriptions[best_match_idx]
            best_score = float(similarities[best_match_idx])
            
            # 색상 추출 (색상 관련 텍스트만 필터링)
            color_keywords = ["빨간색", "파란색", "검은색", "흰색", "회색", "노란색", "초록색", "분홍색"]
            color_matches = {k: text_matches.get(k, 0.0) for k in color_keywords if k in text_matches}
            dominant_color = max(color_matches.items(), key=lambda x: x[1])[0] if color_matches else "알 수 없음"
            
            return {
                "style": best_style,
                "color": dominant_color,
                "pattern": "알 수 없음",  # CLIP으로는 패턴 추출이 어려움
                "text_matches": text_matches,
                "confidence": best_score
            }
            
        except Exception as e:
            print(f"CLIP 분석 오류: {e}")
            # 오류 발생 시 기본값 반환
            return {
                "style": "알 수 없음",
                "color": "알 수 없음",
                "pattern": "알 수 없음",
                "text_matches": {},
                "confidence": 0.0,
                "error": str(e)
            }

class WeatherBasedRecommender:
    """날씨 기반 코디 추천 클래스"""
    
    def __init__(self):
        pass
    
    def get_weather_recommendation(self, temperature, weather, season):
        """날씨와 계절에 맞는 코디 추천"""
        if temperature < 5:
            return {"type": "겨울 코디", "items": ["코트", "스웨터", "부츠"], "layer": "다층"}
        elif temperature < 15:
            return {"type": "가을/봄 코디", "items": ["재킷", "니트", "스니커즈"], "layer": "중간"}
        else:
            return {"type": "여름 코디", "items": ["티셔츠", "반바지", "샌들"], "layer": "단일"}

class MBTIAnalyzer:
    """MBTI 기반 개인화 추천 클래스"""
    
    def __init__(self):
        self.mbti_styles = {
            "ENFP": "자유롭고 컬러풀한 캐주얼 스타일",
            "ISTJ": "깔끔하고 단정한 포멀 스타일",
            "ESFP": "트렌디하고 화려한 스타일",
            "INTJ": "미니멀하고 세련된 스타일"
        }
    
    def get_personality_style(self, mbti_type):
        """MBTI에 맞는 스타일 반환"""
        return self.mbti_styles.get(mbti_type, "균형잡힌 스타일")

class TextBasedSearcher:
    """텍스트 기반 코디 검색 클래스 (CLIP 활용)"""
    
    def __init__(self, clip_analyzer=None):
        """CLIP 분석기를 주입받거나 새로 생성"""
        self.clip_analyzer = clip_analyzer
        self.outfit_categories = {
            "파티용": ["화려한 드레스", "시퀸 원피스", "스팽글 액세서리"],
            "출근룩": ["정장", "블라우스", "슬랙스", "로퍼"],
            "데이트룩": ["로맨틱 원피스", "부드러운 컬러", "우아한 액세서리"]
        }
    
    def search_outfits(self, query, reference_images=None):
        """텍스트 쿼리로 코디 검색 (CLIP 활용)"""
        # 기본 키워드 매칭
        matched_category = None
        for category in self.outfit_categories.keys():
            if category in query:
                matched_category = category
                break
        
        # CLIP을 사용한 이미지-텍스트 매칭 (이미지가 있는 경우)
        if reference_images and self.clip_analyzer:
            # 각 이미지에 대해 텍스트 쿼리와의 유사도 계산
            best_matches = []
            for img in reference_images:
                try:
                    result = self.clip_analyzer.analyze_style(img, [query])
                    if result.get("confidence", 0) > 0.1:
                        best_matches.append({
                            "image": img,
                            "similarity": result.get("confidence", 0),
                            "style": result.get("style", "")
                        })
                except Exception as e:
                    print(f"이미지 분석 오류: {e}")
                    continue
            
            if best_matches:
                # 유사도가 높은 순으로 정렬
                best_matches.sort(key=lambda x: x["similarity"], reverse=True)
                return {
                    "category": matched_category or "일반",
                    "items": self.outfit_categories.get(matched_category, ["캐주얼 웨어"]),
                    "matched": True,
                    "clip_results": best_matches[:3]  # 상위 3개만 반환
                }
        
        # 키워드 매칭 결과 반환
        return {
            "category": matched_category or "일반",
            "items": self.outfit_categories.get(matched_category, ["캐주얼 웨어"]),
            "matched": matched_category is not None
        }

class FashionRecommender:
    """통합 패션 코디 추천 시스템"""
    
    def __init__(self):
        """모든 추천 시스템 컴포넌트 초기화"""
        print("패션 추천 시스템 초기화 중...")
        self.detector = YOLODetector()
        self.analyzer = CLIPAnalyzer()
        self.weather_recommender = WeatherBasedRecommender()
        self.mbti_analyzer = MBTIAnalyzer()
        self.text_searcher = TextBasedSearcher(clip_analyzer=self.analyzer)
        print("패션 추천 시스템 초기화 완료!")
    
    def recommend_outfit(self, image, mbti, temperature, weather, season):
        """통합 코디 추천 파이프라인"""
        # 1. YOLOv5로 옷 아이템 탐지
        detected_items = self.detector.detect_clothes(image)
        
        # 2. CLIP으로 스타일 및 색상 분석
        style_descriptions = ["캐주얼", "포멀", "트렌디", "스포츠", "빈티지", "모던"]
        color_descriptions = ["빨간색", "파란색", "검은색", "흰색", "회색", "갈색", "베이지"]
        all_descriptions = style_descriptions + color_descriptions
        style_analysis = self.analyzer.analyze_style(image, all_descriptions)
        
        # 3. 날씨/계절 정보 고려
        weather_rec = self.weather_recommender.get_weather_recommendation(temperature, weather, season)
        
        # 4. MBTI 개인화 적용
        mbti_style = self.mbti_analyzer.get_personality_style(mbti)
        
        # 5. 탐지된 아이템 기반 추천 생성
        outfit_combinations = []
        
        # 스타일별 추천 생성
        for style in style_descriptions:
            if style in style_analysis.get("text_matches", {}):
                confidence = style_analysis["text_matches"][style]
                if confidence > 0.1:  # 유의미한 유사도만
                    outfit_combinations.append({
                        "style": style,
                        "items": weather_rec["items"],
                        "confidence": confidence,
                        "detected_items": detected_items["items"][:3] if detected_items["items"] else []
                    })
        
        # 추천이 적으면 기본 추천 추가
        if len(outfit_combinations) < 3:
            for style in ["캐주얼", "포멀", "트렌디"]:
                if not any(oc["style"] == style for oc in outfit_combinations):
                    outfit_combinations.append({
                        "style": style,
                        "items": weather_rec["items"],
                        "confidence": 0.5,
                        "detected_items": []
                    })
        
        # confidence 기준으로 정렬
        outfit_combinations.sort(key=lambda x: x["confidence"], reverse=True)
        outfit_combinations = outfit_combinations[:3]  # 상위 3개만
        
        return {
            "detected_items": detected_items,
            "style_analysis": style_analysis,
            "weather_recommendation": weather_rec,
            "mbti_style": mbti_style,
            "outfit_combinations": outfit_combinations
        }
