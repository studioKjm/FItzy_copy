"""
얼굴 및 체형 분석 유틸리티
MediaPipe + DeepFace를 사용한 얼굴 특징, 체형 분석
"""

import numpy as np
from PIL import Image
import cv2
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ mediapipe 라이브러리가 없습니다. pip install mediapipe로 설치하세요.")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, Exception) as e:
    DEEPFACE_AVAILABLE = False
    # 의존성 충돌 등으로 인한 에러는 조용히 처리 (MediaPipe로 대체)
    # print(f"⚠️ deepface 라이브러리를 사용할 수 없습니다: {str(e)}")
    pass


class BodyAnalyzer:
    """얼굴 및 체형 분석 클래스"""
    
    def __init__(self):
        self.deepface_available = DEEPFACE_AVAILABLE
        
        if not MEDIAPIPE_AVAILABLE:
            self.face_mesh = None
            self.pose = None
            return
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection  # Face Detection 추가
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face Detection (Face Mesh보다 탐지율 높음)
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # full range model (전신샷 대응)
            min_detection_confidence=0.1  # 매우 낮은 임계값
        )
        
        # Face Mesh (얼굴 특징 분석용)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,  # 여러 얼굴 감지 허용
            refine_landmarks=True,
            min_detection_confidence=0.05,  # 매우 낮은 임계값 (전신샷 대응)
            min_tracking_confidence=0.05
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # 속도 최적화 (0-2)
            enable_segmentation=False,  # 세그멘테이션 비활성화 (속도 향상)
            min_detection_confidence=0.3  # 감지 민감도 조정
        )
    
    def analyze_face(self, image: Image.Image):
        """얼굴 특징 분석 (MediaPipe 우선, DeepFace는 선택적)"""
        # DeepFace는 버전 충돌 가능성이 있어 MediaPipe 우선 사용
        deepface_result = None
        if self.deepface_available:
            try:
                # PIL Image를 numpy array로 변환
                img_array = np.array(image.convert('RGB'))
                # DeepFace 분석 (성별, 나이, 감정 등)
                # enforce_detection=False: 얼굴을 찾지 못해도 에러 발생하지 않음
                # silent=True: 경고 메시지 억제
                # backend='opencv': OpenCV 백엔드 사용 (더 안정적)
                deepface_result = DeepFace.analyze(
                    img_array,
                    actions=['gender', 'age', 'emotion'],
                    enforce_detection=False,
                    silent=True,
                    detector_backend='opencv'  # 더 안정적인 백엔드
                )
            except Exception as e:
                # DeepFace 실패 시 MediaPipe로 fallback (에러 무시)
                deepface_result = None
        
        # MediaPipe 분석 (기존 로직 유지)
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            # DeepFace 결과만 사용
            if deepface_result:
                return self._parse_deepface_result(deepface_result, image)
            return {
                "detected": False,
                "error": "MediaPipe와 DeepFace를 사용할 수 없습니다."
            }
        
        try:
            # PIL Image를 numpy array로 변환
            img_array = np.array(image.convert('RGB'))
            
            # 이미지 크기 조정 (전신샷 대응: 큰 이미지는 리사이즈)
            max_size = 1000
            height, width = img_array.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 먼저 Face Detection 시도 (Face Mesh보다 탐지율 높음)
            detection_results = self.face_detection.process(rgb_image)
            face_detected = False
            
            if detection_results.detections and len(detection_results.detections) > 0:
                face_detected = True
                # Face Detection 성공 시 Face Mesh로 상세 분석
                mesh_results = self.face_mesh.process(rgb_image)
            else:
                # Face Detection 실패 시 리사이즈 후 재시도
                if max(height, width) > 800:
                    try:
                        resize_size = 800
                        scale = resize_size / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
                        detection_results = self.face_detection.process(resized_rgb)
                        if detection_results.detections and len(detection_results.detections) > 0:
                            face_detected = True
                            mesh_results = self.face_mesh.process(resized_rgb)
                    except:
                        pass
                
                if not face_detected:
                    # Face Mesh로 직접 시도
                    mesh_results = self.face_mesh.process(rgb_image)
                    if mesh_results.multi_face_landmarks:
                        face_detected = True
            
            # 얼굴이 탐지되지 않은 경우
            if not face_detected:
                # mesh_results가 정의되지 않았을 수 있으므로 확인
                if 'mesh_results' not in locals():
                    class EmptyResult:
                        multi_face_landmarks = None
                    mesh_results = EmptyResult()
                
                has_mesh = mesh_results.multi_face_landmarks is not None
                has_detection = hasattr(detection_results, 'detections') and detection_results.detections and len(detection_results.detections) > 0
                
                if not has_mesh and not has_detection:
                    return {
                        "detected": False, 
                        "message": "얼굴을 찾을 수 없습니다. 얼굴이 명확하게 보이는 사진을 업로드해주세요.",
                        "hint": "전신샷의 경우 얼굴이 작아 탐지가 어려울 수 있습니다. 상체 사진을 권장합니다."
                    }
            
            # Face Mesh 결과가 있으면 상세 분석 수행
            results = mesh_results
            if not results.multi_face_landmarks:
                # Face Mesh는 없지만 Detection은 성공한 경우: 얼굴 영역을 확대하여 재시도
                if detection_results.detections and len(detection_results.detections) > 0:
                    detection = detection_results.detections[0]  # 첫 번째 얼굴 사용
                    bbox = detection.location_data.relative_bounding_box
                    
                    # 얼굴 영역 추출 (여유 공간 포함: 20% 확장)
                    h, w = rgb_image.shape[:2]
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # 얼굴 영역 확대 (1.5배)
                    expand_ratio = 1.5
                    center_x = x + width // 2
                    center_y = y + height // 2
                    new_width = int(width * expand_ratio)
                    new_height = int(height * expand_ratio)
                    new_x = max(0, center_x - new_width // 2)
                    new_y = max(0, center_y - new_height // 2)
                    new_x2 = min(w, new_x + new_width)
                    new_y2 = min(h, new_y + new_height)
                    
                    # 얼굴 영역 잘라내기 및 확대
                    face_crop = rgb_image[new_y:new_y2, new_x:new_x2]
                    if face_crop.size > 0:
                        # 얼굴 영역을 최소 300x300으로 확대 (Face Mesh는 작은 얼굴에서도 동작)
                        min_size = 300
                        if max(face_crop.shape[:2]) < min_size:
                            scale = min_size / max(face_crop.shape[:2])
                            new_w = int(face_crop.shape[1] * scale)
                            new_h = int(face_crop.shape[0] * scale)
                            face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        
                        # 확대된 얼굴 영역에서 Face Mesh 재시도
                        mesh_results_retry = self.face_mesh.process(face_crop)
                        if mesh_results_retry.multi_face_landmarks:
                            results = mesh_results_retry
                            rgb_image = face_crop  # 좌표 계산 시 사용할 이미지 변경
            
            if not results.multi_face_landmarks:
                # Face Mesh 최종 실패 시 기본 정보만 반환
                return {
                    "detected": True,
                    "face_shape": "감지됨 (상세 분석 불가)",
                    "eye_size": "알 수 없음",
                    "message": "얼굴은 탐지되었지만 상세 분석을 위해 더 명확한 얼굴 사진이 필요합니다."
                }
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # 얼굴 특징 추출
            # 얼굴 형태 (얼굴 비율 기반 + 추가 특징)
            landmarks = face_landmarks.landmark
            
            # 얼굴 너비/높이 비율 계산
            face_width = abs(landmarks[234].x - landmarks[454].x)  # 좌우 끝
            face_height = abs(landmarks[10].y - landmarks[152].y)  # 상하 끝
            face_ratio = face_width / face_height if face_height > 0 else 1.0
            
            # 추가 특징: 이마 너비 / 턱 너비 비율 (사각형 판별용)
            # 이마 너비 (눈썹 상단)
            forehead_width = abs(landmarks[10].x - landmarks[152].x)  # 양쪽 이마 끝
            # 턱 너비
            jaw_width = abs(landmarks[172].x - landmarks[397].x)  # 턱 양쪽 끝
            jaw_ratio = jaw_width / forehead_width if forehead_width > 0 else 1.0
            
            # 얼굴 하단 비율 (턱선 각도 판별용)
            chin_point = landmarks[175].y  # 턱 끝
            cheek_point = (landmarks[234].y + landmarks[454].y) / 2  # 양쪽 볼 중간
            face_bottom_ratio = abs(chin_point - cheek_point) / face_height if face_height > 0 else 0
            
            # 얼굴 형태 분류 (개선된 로직)
            # 사각형: 이마와 턱 너비가 비슷하고, 턱선이 각져있음
            if 0.85 <= jaw_ratio <= 1.15 and face_bottom_ratio < 0.15:
                face_shape = "사각형"
            # 길쭉한형: 얼굴이 매우 길고 (ratio < 0.70) 턱이 좁음
            elif face_ratio < 0.70:
                face_shape = "길쭉한형"
            # 둥근형: 얼굴이 넓고 (ratio > 0.85) 턱이 둥글음
            elif face_ratio > 0.88:
                face_shape = "둥근형"
            # 계란형: 중간 비율
            elif 0.75 <= face_ratio <= 0.85:
                face_shape = "계란형"
            # 길쭉한형 (경계선)
            else:
                face_shape = "길쭉한형" if face_ratio < 0.75 else "계란형"
            
            # 눈 크기 (대략적)
            left_eye_width = abs(landmarks[33].x - landmarks[133].x)
            right_eye_width = abs(landmarks[362].x - landmarks[263].x)
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            
            # DeepFace 결과와 결합
            result = {
                "detected": True,
                "face_shape": face_shape,
                "face_ratio": float(face_ratio),
                "eye_size": "큰 편" if avg_eye_width > 0.05 else "작은 편",
                "landmarks_count": len(landmarks)
            }
            
            # DeepFace 결과가 있으면 추가 정보 병합
            if deepface_result:
                result.update(self._parse_deepface_result(deepface_result, image, merge=True))
            
            return result
            
        except Exception as e:
            # DeepFace 결과가 있으면 사용
            if deepface_result:
                return self._parse_deepface_result(deepface_result, image)
            return {
                "detected": False,
                "error": str(e)
            }
    
    def _parse_deepface_result(self, deepface_result, image: Image.Image, merge=False):
        """DeepFace 결과 파싱"""
        result = {} if merge else {"detected": True}
        
        # DeepFace 결과는 리스트 또는 딕셔너리일 수 있음
        if isinstance(deepface_result, list):
            deepface_result = deepface_result[0] if deepface_result else {}
        
        if isinstance(deepface_result, dict):
            # 성별 정보
            gender = deepface_result.get('dominant_gender', '')
            if gender:
                result['gender_deepface'] = "남성" if "Man" in gender or "Male" in gender else "여성"
            
            # 나이 정보
            age = deepface_result.get('age', None)
            if age:
                result['age'] = int(age)
            
            # 감정 정보
            emotion = deepface_result.get('dominant_emotion', '')
            if emotion:
                result['emotion'] = emotion
            
            # 얼굴 영역 정보 (bbox)
            region = deepface_result.get('region', {})
            if region:
                result['face_bbox'] = region
        
        # MediaPipe 결과가 없으면 기본값 설정
        if not merge:
            if 'face_shape' not in result:
                result['face_shape'] = "감지됨 (상세 분석 불가)"
            if 'eye_size' not in result:
                result['eye_size'] = "알 수 없음"
        
        return result
    
    def detect_gender(self, image: Image.Image):
        """이미지에서 성별 자동 인식 (얼굴 특징 기반 추정 + DeepFace 선택적)"""
        # 방법 1: DeepFace 사용 (가장 정확, 버전 충돌 방지를 위해 선택적)
        if self.deepface_available:
            try:
                img_array = np.array(image.convert('RGB'))
                result = DeepFace.analyze(
                    img_array,
                    actions=['gender'],
                    enforce_detection=False,
                    silent=True,
                    detector_backend='opencv'
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                gender = result.get('dominant_gender', '')
                if gender:
                    if "Man" in gender or "Male" in gender:
                        return "남성"
                    elif "Woman" in gender or "Female" in gender:
                        return "여성"
            except Exception:
                pass
        
        # 방법 2: MediaPipe 얼굴 특징 기반 성별 추정 (휴리스틱)
        if MEDIAPIPE_AVAILABLE and self.face_mesh:
            try:
                # 얼굴 분석 결과 가져오기
                face_info = self.analyze_face(image)
                
                if face_info and face_info.get("detected"):
                    # 얼굴 특징 기반 성별 추정
                    gender = self._estimate_gender_from_features(face_info)
                    if gender:
                        return gender
            except Exception:
                pass
        
        return None
    
    def _estimate_gender_from_features(self, face_info: dict) -> str:
        """얼굴 특징을 기반으로 성별 추정 (휴리스틱 - 정확도 제한적)"""
        if not face_info.get("detected"):
            return None
        
        # 얼굴 특징 추출
        face_ratio = face_info.get("face_ratio", 1.0)
        face_shape = face_info.get("face_shape", "")
        
        # 연구 기반 휴리스틱 (통계적 경향성)
        # 주의: 이것은 정확도가 낮은 추정 방법임 (60-70% 정도)
        
        score_male = 0
        score_female = 0
        
        # 얼굴 비율 분석 (남성은 일반적으로 더 각진 경향)
        if face_ratio < 0.72:
            score_male += 2  # 길쭉한 얼굴 (남성 경향)
        elif face_ratio > 0.88:
            score_female += 2  # 둥근 얼굴 (여성 경향)
        elif 0.75 <= face_ratio <= 0.85:
            # 이상적 비율 (양쪽 모두 가능)
            pass
        
        # 얼굴 형태 분석
        if face_shape == "사각형":
            score_male += 3  # 각진 형태 (남성 경향)
        elif face_shape == "둥근형":
            score_female += 3  # 둥근 형태 (여성 경향)
        elif face_shape == "계란형":
            score_female += 1  # 계란형은 여성에게 더 흔함
        
        # 눈 크기 (여성이 일반적으로 눈이 더 큰 경향)
        eye_size = face_info.get("eye_size", "")
        if eye_size == "큰 편":
            score_female += 1
        
        # 판단 (명확한 차이가 있을 때만)
        if score_male > score_female + 2:
            return "남성"
        elif score_female > score_male + 2:
            return "여성"
        
        # 불확실한 경우 None 반환 (다른 방법에 의존)
        return None
    
    def analyze_body(self, image: Image.Image):
        """체형 분석"""
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return {
                "detected": False,
                "error": "MediaPipe를 사용할 수 없습니다."
            }
        
        try:
            img_array = np.array(image.convert('RGB'))
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 포즈 분석
            results = self.pose.process(rgb_image)
            
            if not results.pose_landmarks:
                return {"detected": False, "message": "체형을 분석할 수 없습니다."}
            
            landmarks = results.pose_landmarks.landmark
            
            # 키 포인트 추출
            def get_point(idx):
                if idx < len(landmarks):
                    return landmarks[idx].x, landmarks[idx].y
                return None
            
            # 어깨 너비
            left_shoulder = get_point(11)  # 왼쪽 어깨
            right_shoulder = get_point(12)  # 오른쪽 어깨
            
            # 엉덩이 너비
            left_hip = get_point(23)
            right_hip = get_point(24)
            
            # 체형 비율 계산 (정규화된 좌표 사용)
            shoulder_width = None
            hip_width = None
            body_ratio = None
            
            if left_shoulder and right_shoulder:
                shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            
            if left_hip and right_hip:
                hip_width = abs(left_hip[0] - right_hip[0])
            
            if shoulder_width and hip_width and hip_width > 0:
                body_ratio = shoulder_width / hip_width
            else:
                body_ratio = None
            
            # 체형 분류 (개선된 로직)
            body_type = "균형잡힌 체형"  # 기본값
            if body_ratio is not None:
                # 실제 체형 분석: 어깨/힙 비율은 보통 0.9~1.1 범위가 정상
                # MediaPipe 좌표는 정규화되어 있으므로, 실제 비율보다 크게 나타날 수 있음
                # 따라서 더 보수적인 임계값 사용
                
                # 통계적 분석: 일반적인 체형 비율
                # - 균형잡힌 체형: 0.85 ~ 1.15
                # - 어깨가 넓은 체형: 1.15 이상 (명확히 넓어야 함)
                # - 힙이 넓은 체형: 0.85 미만 (명확히 좁아야 함)
                
                if body_ratio > 1.20:  # 1.15에서 1.20으로 더 엄격하게 (20% 이상 차이)
                    body_type = "어깨가 넓은 체형"
                elif body_ratio < 0.80:  # 0.85에서 0.80으로 더 엄격하게 (20% 이상 차이)
                    body_type = "힙이 넓은 체형"
                else:
                    body_type = "균형잡힌 체형"
            else:
                # 키 포인트 부족 시 기본값
                body_type = "분석 불가"
            
            # 키 추정 (대략적, 이미지 비율 기반)
            height_ratio = None
            head = get_point(0)  # 코
            if head and left_hip:
                height_ratio = abs(head[1] - left_hip[1])
            
            return {
                "detected": True,
                "body_type": body_type,
                "body_ratio": float(body_ratio) if body_ratio else None,
                "shoulder_width_ratio": float(shoulder_width) if shoulder_width else None,
                "hip_width_ratio": float(hip_width) if hip_width else None,
                "height_ratio": float(height_ratio) if height_ratio else None
            }
            
        except Exception as e:
            return {
                "detected": False,
                "error": str(e)
            }
    
    def get_recommendation_based_on_body(self, face_info: dict, body_info: dict):
        """체형 기반 추천 로직"""
        recommendations = []
        
        if not face_info.get("detected") and not body_info.get("detected"):
            return recommendations
        
        # 얼굴 형태 기반
        if face_info.get("detected"):
            face_shape = face_info.get("face_shape", "")
            if face_shape == "둥근형":
                recommendations.append("V넥이나 U넥으로 얼굴을 길게 보이게")
            elif face_shape == "길쭉한형":
                recommendations.append("둥근넥이나 터틀넥으로 균형 잡기")
        
        # 체형 기반
        if body_info.get("detected"):
            body_type = body_info.get("body_type", "")
            if "어깨가 넓은" in body_type:
                recommendations.append("V넥 상의로 어깨 라인 부드럽게")
                recommendations.append("하의는 A라인으로 균형 잡기")
            elif "힙이 넓은" in body_type:
                recommendations.append("상의는 밝은색으로 상체 강조")
                recommendations.append("하의는 다크톤으로 하체 라인 조절")
            elif "균형잡힌" in body_type:
                recommendations.append("균형잡힌 체형이니 다양한 스타일 가능")
        
        return recommendations

