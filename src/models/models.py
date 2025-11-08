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
    
    # 영어 클래스 이름 → 한국어 매핑 (DeepFashion2 13개 클래스)
    FASHION_CLASS_MAPPING = {
        "long sleeve dress": "긴팔 드레스",
        "long sleeve outwear": "긴팔 아우터",
        "long sleeve top": "긴팔 상의",
        "short sleeve dress": "반팔 드레스",
        "short sleeve outwear": "반팔 아우터",
        "short sleeve top": "반팔 상의",
        "shorts": "반바지",
        "skirt": "스커트",
        "sling dress": "끈 드레스",
        "sling": "끈 상의",
        "trousers": "바지",
        "vest dress": "조끼 드레스",
        "vest": "조끼"
    }
    
    def __init__(self, model_path=None):
        """YOLOv5 모델 초기화"""
        if model_path is None:
            model_path = YOLO_MODEL_PATH
        
        # 디바이스 설정 (메타 텐서 문제 방지 - CPU 우선 사용)
        # ultralytics 내부적으로 device를 처리하므로 명시적으로 지정
        device = "cpu"  # 메타 텐서 문제 방지를 위해 CPU 사용 (나중에 필요시 변경 가능)
        
        # 모델 파일이 없으면 사전 학습된 모델 사용 (yolov5n, yolov5s 등)
        if not os.path.exists(model_path):
            print(f"모델 파일이 없습니다: {model_path}")
            print("사전 학습된 YOLOv5 모델을 사용합니다: yolov5n")
            # COCO 사전 학습 모델 사용 (person, bag 등 일반 객체 탐지)
            try:
                # device를 명시적으로 지정하여 모델 로드
                self.model = YOLO('yolov5n.pt')
                # ultralytics는 내부적으로 device를 처리하므로 .to() 호출하지 않음
            except Exception as e:
                print(f"⚠️ 모델 로드 중 오류 발생: {e}")
                print("💡 에러가 지속되면 앱을 재시작하세요.")
                raise
            self.is_fashion_model = False
            print("일반 객체 탐지 모델로 동작합니다. 패션 전용 모델 학습이 필요합니다.")
        else:
            try:
                # 패션 모델 로드
                # ultralytics YOLO는 체크포인트를 로드할 때 device를 자동으로 처리
                # 메타 텐서 문제를 피하기 위해 모델을 직접 로드하지 않고 
                # ultralytics의 내장 로딩 방식을 신뢰
                self.model = YOLO(model_path)
                # 모델이 완전히 로드되면 내부 모델 객체에 접근
                # device 이동은 ultralytics가 자동으로 처리
            except NotImplementedError as meta_error:
                # 메타 텐서 오류인 경우 특별 처리
                if "meta tensor" in str(meta_error).lower():
                    print(f"⚠️ 메타 텐서 문제 감지: {meta_error}")
                    print("💡 모델 체크포인트를 다시 다운로드하거나 학습된 모델을 확인하세요.")
                    print("💡 임시로 사전 학습된 모델을 사용합니다...")
                    self.model = YOLO('yolov5n.pt')
                    self.is_fashion_model = False
                    return
                else:
                    raise
            except Exception as e:
                print(f"⚠️ 패션 모델 로드 실패: {e}")
                print("💡 사전 학습된 모델로 대체...")
                try:
                    self.model = YOLO('yolov5n.pt')
                    self.is_fashion_model = False
                except Exception as e2:
                    print(f"⚠️ 사전 학습 모델 로드도 실패: {e2}")
                    raise
                return
            
            self.is_fashion_model = True
            print(f"YOLOv5 패션 모델 로드 완료: {model_path}")
            # 학습된 클래스 확인
            if hasattr(self.model, 'names') and self.model.names:
                classes_list = list(self.model.names.values())
                print(f"탐지 가능한 클래스: {classes_list[:5]}...")
    
    def detect_clothes(self, image, clip_analyzer=None):
        """이미지에서 옷 아이템 탐지 (CLIP 검증 포함)"""
        # 이미지 전처리
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            pil_image = image
        elif isinstance(image, np.ndarray):
            img_array = image
            pil_image = Image.fromarray(img_array)
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
                
                # 클래스 이름 가져오기
                class_name = self.model.names[class_id]
                
                # 패션 모델인 경우: 모든 탐지 결과 사용 (이미 패션 아이템만 탐지)
                if self.is_fashion_model:
                    # 신뢰도 임계값 설정 (상향 조정)
                    if confidence > 0.3:  # 임계값 상향 (0.25 → 0.3)
                        # 한국어 클래스 이름으로 변환
                        korean_name = self.FASHION_CLASS_MAPPING.get(class_name, class_name)
                        
                        # CLIP 검증 (긴팔/반팔 구분 등)
                        verified_class = self._verify_detection_with_clip(
                            pil_image, bbox, class_name, korean_name, clip_analyzer
                        ) if clip_analyzer else (class_name, korean_name)
                        
                        verified_class_en, verified_class_ko = verified_class
                        
                        detected_items.append({
                            "class": verified_class_ko,
                            "class_en": verified_class_en,
                            "confidence": confidence,
                            "bbox": bbox,
                            "original_class": class_name  # 원본 클래스도 저장 (디버깅용)
                        })
                else:
                    # COCO 모델인 경우: 기존 필터링 로직 유지
                    fashion_related = ['person', 'handbag', 'backpack', 'suitcase', 'sports ball']
                    if class_name in fashion_related or confidence > 0.3:
                        detected_items.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": bbox
                        })
        
        return {
            "items": detected_items,
            "image_size": image.size if isinstance(image, Image.Image) else (img_array.shape[1], img_array.shape[0]),
            "is_fashion_model": self.is_fashion_model
        }
    
    def _verify_detection_with_clip(self, image, bbox, class_name, korean_name, clip_analyzer):
        """CLIP을 사용하여 YOLO 탐지 결과 검증 (특히 긴팔/반팔 구분)"""
        try:
            # bbox로 이미지 영역 잘라내기
            x1, y1, x2, y2 = map(int, bbox)
            width, height = image.size
            
            # bbox 범위 확인
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(x1, min(x2, width))
            y2 = max(y1, min(y2, height))
            
            if x2 - x1 < 10 or y2 - y1 < 10:  # 너무 작은 영역은 건너뛰기
                return (class_name, korean_name)
            
            # 의상 영역 추출
            crop_image = image.crop((x1, y1, x2, y2))
            
            # 긴팔/반팔 구분 검증
            if "sleeve" in class_name.lower() or "상의" in korean_name:
                # 긴팔 vs 반팔 검증
                long_sleeve_keywords = ["긴팔", "long sleeve", "롱 슬리브", "소매가 긴"]
                short_sleeve_keywords = ["반팔", "short sleeve", "숏 슬리브", "소매가 짧은", "팔이 보이는"]
                
                # CLIP으로 실제 팔 길이 확인
                test_keywords = ["long sleeve shirt", "short sleeve shirt", "긴팔", "반팔"]
                similarity_result = clip_analyzer.analyze_style(crop_image, test_keywords)
                
                if similarity_result and similarity_result.get("text_matches"):
                    matches = similarity_result["text_matches"]
                    long_score = sum(v for k, v in matches.items() if any(word in k.lower() for word in ["long", "긴"]))
                    short_score = sum(v for k, v in matches.items() if any(word in k.lower() for word in ["short", "반"]))
                    
                    # CLIP 검증 결과가 YOLO 결과와 다르면 수정
                    is_originally_long = "long" in class_name.lower() or "긴팔" in korean_name
                    
                    if short_score > long_score + 0.2 and is_originally_long:
                        # 반팔로 수정
                        if "top" in class_name:
                            return ("short sleeve top", "반팔 상의")
                        elif "outwear" in class_name:
                            return ("short sleeve outwear", "반팔 아우터")
                        elif "dress" in class_name:
                            return ("short sleeve dress", "반팔 드레스")
                    elif long_score > short_score + 0.2 and not is_originally_long:
                        # 긴팔로 수정
                        if "top" in class_name:
                            return ("long sleeve top", "긴팔 상의")
                        elif "outwear" in class_name:
                            return ("long sleeve outwear", "긴팔 아우터")
                        elif "dress" in class_name:
                            return ("long sleeve dress", "긴팔 드레스")
        except Exception as e:
            # 검증 실패 시 원본 클래스 반환
            pass
        
        return (class_name, korean_name)

class CLIPAnalyzer:
    """CLIP 모델을 사용한 스타일 분석 클래스"""
    
    def detect_gender(self, image):
        """CLIP을 사용한 성별 인식 (개선: 더 구체적인 키워드 사용)"""
        try:
            # 더 구체적인 성별 관련 키워드
            gender_texts = [
                "남성 패션", "여성 패션", "남자 옷", "여자 옷",
                "male clothing", "female clothing", "men's fashion", "women's fashion",
                "남성 의상", "여성 의상", "남성 스타일", "여성 스타일"
            ]
            similarities = self.analyze_style(image, gender_texts)
            
            if similarities and similarities.get("text_matches"):
                matches = similarities["text_matches"]
                # 남성 관련 키워드 점수 합산
                male_score = sum(v for k, v in matches.items() if any(word in k.lower() for word in ["남성", "남자", "male", "men"]))
                # 여성 관련 키워드 점수 합산
                female_score = sum(v for k, v in matches.items() if any(word in k.lower() for word in ["여성", "여자", "female", "women"]))
                
                # 임계값 상향: 더 확실할 때만 판별
                score_diff = abs(male_score - female_score)
                if male_score > female_score and score_diff > 0.15:  # 0.15 이상 차이
                    return "남성"
                elif female_score > male_score and score_diff > 0.15:
                    return "여성"
                else:
                    return None  # 불확실하면 None 반환 (의상 기반 판단에 의존)
            return None
        except Exception as e:
            return None
    
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
        # 성별별 아이템 카테고리
        self.outfit_categories = {
            "파티용": {
                "남성": ["화려한 정장", "시퀸 재킷", "스팽글 액세서리", "정장화"],
                "여성": ["화려한 드레스", "시퀸 원피스", "스팽글 액세서리"]
            },
            "출근룩": {
                "남성": ["정장 재킷", "셔츠", "슬랙스", "로퍼"],
                "여성": ["정장 재킷", "블라우스", "슬랙스", "로퍼"]
            },
            "데이트룩": {
                "남성": ["세련된 셔츠", "부드러운 컬러 재킷", "우아한 액세서리"],
                "여성": ["로맨틱 원피스", "부드러운 컬러", "우아한 액세서리"]
            }
        }
    
    def search_outfits(self, query, reference_images=None, gender=None):
        """텍스트 쿼리로 코디 검색 (CLIP 활용)"""
        # 기본 키워드 매칭
        matched_category = None
        for category in self.outfit_categories.keys():
            if category in query:
                matched_category = category
                break
        
        # 성별 기본값 설정 (전달되지 않은 경우)
        if gender is None:
            gender = "여성"  # 기본값 (하위 호환성)
        
        # 성별에 맞는 아이템 가져오기
        items = ["캐주얼 웨어"]  # 기본값
        if matched_category and matched_category in self.outfit_categories:
            category_items = self.outfit_categories[matched_category]
            items = category_items.get(gender, category_items.get("여성", ["캐주얼 웨어"]))
        
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
                    "items": items,
                    "matched": True,
                    "clip_results": best_matches[:3]  # 상위 3개만 반환
                }
        
        # 키워드 매칭 결과 반환
        return {
            "category": matched_category or "일반",
            "items": items,
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
        # 1. YOLOv5로 옷 아이템 탐지 (CLIP 검증 포함)
        detected_items = self.detector.detect_clothes(image, clip_analyzer=self.analyzer)
        
        # 2. CLIP으로 스타일 및 색상 분석
        style_descriptions = ["캐주얼", "포멀", "트렌디", "스포츠", "빈티지", "모던"]
        # 색상 키워드 확장 (한국어 + 영어)
        color_descriptions = [
            "빨간색", "파란색", "검은색", "흰색", "회색", "갈색", "베이지",
            "노란색", "옐로우", "yellow",  # 추가
            "보라색", "퍼플", "purple",
            "오렌지", "주황색", "orange",
            "초록색", "그린", "green",
            "분홍색", "핑크", "pink",
            "네이비", "navy",
            "카키", "khaki",
            "white", "black", "red", "blue"  # 영어 기본 색상
        ]
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
