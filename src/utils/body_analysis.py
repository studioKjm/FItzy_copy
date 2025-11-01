"""
얼굴 및 체형 분석 유틸리티
MediaPipe를 사용한 얼굴 특징, 체형 분석
"""

import numpy as np
from PIL import Image
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ mediapipe 라이브러리가 없습니다. pip install mediapipe로 설치하세요.")


class BodyAnalyzer:
    """얼굴 및 체형 분석 클래스"""
    
    def __init__(self):
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
        """얼굴 특징 분석"""
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            return {
                "detected": False,
                "error": "MediaPipe를 사용할 수 없습니다."
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
                # Face Mesh는 없지만 Detection은 성공한 경우 기본 정보만 반환
                return {
                    "detected": True,
                    "face_shape": "감지됨 (상세 분석 불가)",
                    "eye_size": "알 수 없음",
                    "message": "얼굴은 탐지되었지만 상세 분석을 위해 더 명확한 얼굴 사진이 필요합니다."
                }
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # 얼굴 특징 추출
            # 얼굴 형태 (얼굴 비율 기반)
            landmarks = face_landmarks.landmark
            
            # 얼굴 너비/높이 비율 계산
            face_width = abs(landmarks[234].x - landmarks[454].x)  # 좌우 끝
            face_height = abs(landmarks[10].y - landmarks[152].y)  # 상하 끝
            face_ratio = face_width / face_height if face_height > 0 else 1.0
            
            # 얼굴 형태 분류
            if face_ratio > 0.85:
                face_shape = "둥근형"
            elif face_ratio < 0.75:
                face_shape = "길쭉한형"
            else:
                face_shape = "계란형"
            
            # 눈 크기 (대략적)
            left_eye_width = abs(landmarks[33].x - landmarks[133].x)
            right_eye_width = abs(landmarks[362].x - landmarks[263].x)
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            
            return {
                "detected": True,
                "face_shape": face_shape,
                "face_ratio": float(face_ratio),
                "eye_size": "큰 편" if avg_eye_width > 0.05 else "작은 편",
                "landmarks_count": len(landmarks)
            }
            
        except Exception as e:
            return {
                "detected": False,
                "error": str(e)
            }
    
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
            
            # 체형 분류 (더 엄격한 기준 적용)
            body_type = "균형잡힌 체형"  # 기본값
            if body_ratio is not None:
                # 실제 체형 분석: 어깨/힙 비율은 보통 0.8~1.2 범위
                # 어깨가 넓은 경우: 비율이 1.0보다 훨씬 커야 함
                if body_ratio > 1.15:  # 기존 1.1에서 1.15로 상향
                    body_type = "어깨가 넓은 체형"
                elif body_ratio < 0.85:  # 기존 0.9에서 0.85로 하향
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

