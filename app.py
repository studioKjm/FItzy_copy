"""
Fitzy 패션 코디 추천 앱 - 메인 애플리케이션
Streamlit 기반 웹 인터페이스
"""

import streamlit as st
import datetime
import os
from PIL import Image
import io
from src.utils.recommendation_engine import RecommendationEngine
from src.models.models import FashionRecommender
from src.utils.model_manager import ModelManager
from src.utils.visualization import draw_detections
from src.utils.background_removal import remove_background, extract_person_mask
from src.utils.body_analysis import BodyAnalyzer
from src.utils.scoring_system import ScoringSystem
from config import MBTI_STYLES, SEASONAL_GUIDE, WEATHER_GUIDE

# 전역 변수로 추천 엔진 초기화
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = RecommendationEngine()
if 'fashion_recommender' not in st.session_state:
    st.session_state.fashion_recommender = FashionRecommender()
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'body_analyzer' not in st.session_state:
    st.session_state.body_analyzer = BodyAnalyzer()
if 'scoring_system' not in st.session_state:
    st.session_state.scoring_system = ScoringSystem()

def detect_gender_from_image(image, clip_analyzer, result=None):
    """이미지에서 성별 인식 (의상 기반 + CLIP 조합 - 개선)"""
    detected_gender = None
    
    # 방법 1: 탐지된 의상 기반 판단 (우선순위 높음)
    if result and result.get("detected_items", {}).get("items"):
        items = result["detected_items"]["items"]
        if items:
            classes = []
            for item in items:
                class_ko = item.get("class", "")
                class_en = item.get("class_en", "")
                if class_ko:
                    classes.append(class_ko.lower())
                if class_en:
                    classes.append(class_en.lower())
            
            all_classes_str = " ".join(classes)
            
            # 여성 의상 특징 (더 많은 키워드)
            female_keywords = ["dress", "드레스", "skirt", "스커트", "sling", "끈", 
                              "vest dress", "조끼 드레스", "sling dress", "끈 드레스"]
            # 남성 의상 특징 (더 정확한 키워드)
            male_keywords = ["shirt", "셔츠", "trousers", "바지", "vest", "조끼"]
            
            female_count = sum(1 for kw in female_keywords if kw in all_classes_str)
            male_count = sum(1 for kw in male_keywords if kw in all_classes_str)
            
            # 더 엄격한 판단: 명확한 차이가 있을 때만
            if female_count > 0 and female_count > male_count:
                detected_gender = "여성"
            elif male_count > 0 and male_count > female_count:
                detected_gender = "남성"
    
    # 방법 2: CLIP 기반 인식 (의상 기반이 불확실한 경우만)
    if not detected_gender:
        try:
            clip_gender = clip_analyzer.detect_gender(image)
            if clip_gender:
                detected_gender = clip_gender
        except:
            pass
    
    return detected_gender

def main():
    """메인 애플리케이션 함수"""
    st.title("👗 Fitzy - AI 패션 코디 추천")
    st.markdown("업로드한 옷 이미지로 최적의 코디를 추천받아보세요!")
    
    # 사이드바 - 사용자 설정
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # MBTI 선택
        mbti_type = st.selectbox("MBTI 유형", 
                                ["ENFP", "ISTJ", "ESFP", "INTJ", "기타"])
        
        # 성별 선택 (자동 인식 기능)
        gender_options = ["남성", "여성", "공용"]
        
        # 초기화
        if 'selected_gender' not in st.session_state:
            st.session_state.selected_gender = 0
        
        # rerun 후 자동 업데이트 플래그 확인 및 리셋
        if 'gender_auto_update_pending' in st.session_state and st.session_state.gender_auto_update_pending:
            if 'auto_gender' in st.session_state and st.session_state.auto_gender:
                gender_index_map = {"남성": 0, "여성": 1, "공용": 2}
                auto_index = gender_index_map.get(st.session_state.auto_gender, st.session_state.selected_gender)
                st.session_state.selected_gender = auto_index
            st.session_state.gender_auto_update_pending = False
        
        # 자동 인식된 성별이 있으면 즉시 업데이트
        if 'auto_gender' in st.session_state and st.session_state.auto_gender:
            gender_index_map = {"남성": 0, "여성": 1, "공용": 2}
            auto_index = gender_index_map.get(st.session_state.auto_gender, st.session_state.selected_gender)
            # 자동 인식 성별로 강제 업데이트 (이미지 변경 시 자동 반영)
            if st.session_state.selected_gender != auto_index:
                st.session_state.selected_gender = auto_index
        
        # selectbox: 현재 선택된 성별로 표시
        # key에 성별 인덱스를 포함하여 값이 변경되면 재생성되도록 함
        current_selected_index = st.session_state.selected_gender
        gender = st.selectbox(
            "성별", 
            gender_options, 
            index=current_selected_index,
            key=f"gender_selectbox_{current_selected_index}"  # 인덱스 변경 시 재생성
        )
        
        # 수동 선택 시 업데이트 (사용자가 직접 변경한 경우)
        current_selected_gender = gender_options[current_selected_index]
        if gender != current_selected_gender:
            st.session_state.selected_gender = gender_options.index(gender)
        
        # 자동 인식 성별 표시 (즉시 표시)
        if 'auto_gender' in st.session_state and st.session_state.auto_gender:
            if gender == st.session_state.auto_gender:
                st.success(f"✅ 자동 인식: {st.session_state.auto_gender}")
            else:
                # 자동 인식과 다르면 표시만 (이미지 분석 부분에서 rerun이 처리됨)
                st.info(f"🤖 자동 인식: {st.session_state.auto_gender}")
                # selected_gender는 이미 업데이트되었으므로 rerun 후 반영됨

        # 진단 모드
        debug_mode = st.toggle("🔍 진단 모드 (YOLO/CLIP 상세 분석)", value=False)
        
        # AI 이미지 생성 설정 (선택적)
        with st.expander("🎨 AI 이미지 생성 설정", expanded=False):
            # 초기화 (한 번만)
            if 'enable_ai_images' not in st.session_state:
                st.session_state.enable_ai_images = True
            if 'auto_generate_images' not in st.session_state:
                st.session_state.auto_generate_images = True
            if 'image_gen_method' not in st.session_state:
                st.session_state.image_gen_method = "stable_diffusion"
            if 'num_auto_images' not in st.session_state:
                st.session_state.num_auto_images = 1
            
            # 위젯 표시 (key를 지정하면 자동으로 session_state에 저장됨)
            enable_ai_images = st.toggle(
                "AI 이미지 생성 활성화", 
                value=st.session_state.enable_ai_images, 
                key="enable_ai_images"
            )
            auto_generate = st.toggle(
                "자동 생성 (추천 코디 표시 시 자동 생성)", 
                value=st.session_state.auto_generate_images, 
                key="auto_generate_images"
            )
            
            if enable_ai_images:
                image_gen_method = st.selectbox(
                    "이미지 생성 방법",
                    ["huggingface_api", "dall_e", "stable_diffusion", "stability_ai"],
                    index=0,  # huggingface_api 기본값
                    key="image_gen_method",
                    help="huggingface_api: 무료 (Hugging Face API), dall_e: 유료 (OpenAI), stable_diffusion: 무료 (로컬, GPU 필요), stability_ai: 유료 (Stability AI)"
                )
                
                # 생성할 이미지 개수 선택
                if auto_generate:
                    num_auto_images = st.slider(
                        "자동 생성할 이미지 개수 (추천 코디 중)",
                        min_value=1,
                        max_value=3,
                        value=st.session_state.num_auto_images,
                        key="num_auto_images",
                        help="추천 코디 3개 중 몇 개의 이미지를 자동 생성할지 선택"
                    )
                
                # API 키 입력 (Hugging Face의 경우)
                if image_gen_method == "huggingface_api":
                    hf_api_key = st.text_input(
                        "Hugging Face API 키 (선택적)",
                        value=os.getenv("HUGGINGFACE_API_KEY", ""),
                        type="password",
                        key="hf_api_key_input",
                        help="무료 티어는 API 키 없이도 사용 가능하지만, 키가 있으면 더 빠릅니다. 빈칸으로 두면 무료 티어 사용"
                    )
                    if hf_api_key:
                        # 환경 변수에 임시 설정 (세션 동안만)
                        os.environ["HUGGINGFACE_API_KEY"] = hf_api_key
                    else:
                        # 빈 키면 환경 변수에서도 제거
                        if "HUGGINGFACE_API_KEY" in os.environ:
                            del os.environ["HUGGINGFACE_API_KEY"]
                    
                    if not hf_api_key:
                        st.warning("⚠️ **API 키 필수**: 최근 정책 변경으로 모든 모델에 API 키가 필요합니다.")
                        st.info("💡 무료 계정으로도 API 키 발급 가능합니다.")
                    else:
                        st.success("✅ API 키 설정됨")
                    
                    with st.expander("📖 API 키 발급 방법 (단계별)", expanded=False):
                        st.markdown("""
                        1. **Hugging Face 계정 생성** (무료)
                           - https://huggingface.co/join 접속
                        2. **API 토큰 생성**
                           - https://huggingface.co/settings/tokens 접속
                           - "New token" 클릭
                           - Name: `fitzy-app` (임의)
                           - Type: **"Read"** 선택 (⚠️ 필수!)
                           - "Generate a token" 클릭
                           - 생성된 토큰 복사 (한 번만 표시됨)
                        3. **앱에 입력**
                           - 위 입력란에 복사한 토큰 붙여넣기
                        """)
                    
                    st.caption("🔗 API 키 발급: https://huggingface.co/settings/tokens")
                    st.caption("⚠️ 'Read' 권한 필수! 다른 권한 선택 시 403 오류 발생")
                
                # API 키 안내 (다른 방법들)
                elif image_gen_method == "dall_e":
                    st.info("💡 OpenAI API 키가 필요합니다: 환경 변수 OPENAI_API_KEY 설정")
                elif image_gen_method == "stability_ai":
                    st.info("💡 Stability AI API 키가 필요합니다: 환경 변수 STABILITY_AI_API_KEY 설정")
                elif image_gen_method == "stable_diffusion":
                    st.info("💡 로컬 실행 (M2 맥북 지원)")
                    st.caption("📦 설치: `pip install diffusers accelerate`")
                    st.caption("🍎 Apple Silicon (M1/M2) 자동 감지 및 최적화")
                    st.caption("💾 메모리: 약 4GB 모델 다운로드 필요 (처음만)")
                    st.caption("⏱️ 생성 시간: 약 30-60초 (M2 맥북 기준)")

        # 날씨 정보 입력
        st.subheader("🌤️ 날씨 정보")
        temperature = st.slider("온도 (°C)", -10, 40, 20)
        weather = st.selectbox("날씨", ["맑음", "흐림", "비", "눈", "바람"])
        
        # 계절 선택
        season = st.selectbox("계절", ["봄", "여름", "가을", "겨울"])
    
    # 메인 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📸 이미지 분석", "🔍 텍스트 검색", "🌟 트렌드 코디", "⚙️ 모델 관리"])
    
    with tab1:
        # 이미지 업로드 및 분석
        uploaded_file = st.file_uploader("옷 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'], key="image_uploader")
        
        # 이미지가 변경되었는지 확인하기 위한 키
        if uploaded_file:
            # 파일이 변경되었는지 확인
            file_id = uploaded_file.name + str(uploaded_file.size)
            if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
                st.session_state.last_file_id = file_id
                # 이미지 관련 캐시 초기화
                if 'processed_image' in st.session_state:
                    del st.session_state.processed_image
                if 'face_info_cache' in st.session_state:
                    del st.session_state.face_info_cache
                if 'body_info_cache' in st.session_state:
                    del st.session_state.body_info_cache
            st.success("이미지 업로드 완료! 분석 중...")
            # 이미지 로드
            image = Image.open(uploaded_file)
            
            # 자동 배경 제거 시도
            from src.utils.background_removal import REMBG_AVAILABLE
            processed_image = image
            bg_removed = False
            bg_error = None
            
            if REMBG_AVAILABLE:
                with st.spinner("🎭 배경 제거 중..."):
                    try:
                        processed_image = remove_background(image)
                        # 배경 제거 성공 여부 확인 (RGBA 모드면 성공)
                        if processed_image.mode == 'RGBA':
                            bg_removed = True
                            # 알파 채널이 실제로 있는지 확인
                            alpha = processed_image.split()[3]
                            if alpha.getextrema()[0] < 255:  # 일부라도 투명하면 성공
                                bg_removed = True
                            else:
                                # 모두 불투명하면 배경 제거 실패로 간주
                                bg_removed = False
                                bg_error = "배경 제거 결과가 모두 불투명합니다."
                        else:
                            # RGB 모드면 배경 제거 실패로 간주
                            processed_image = image
                            bg_removed = False
                            bg_error = f"배경 제거 결과가 RGB 모드입니다 (예상: RGBA)"
                    except Exception as e:
                        # 에러 발생 시 원본 이미지 사용
                        processed_image = image
                        bg_removed = False
                        bg_error = f"배경 제거 중 오류: {str(e)}"
            else:
                # rembg가 없으면 원본 이미지 사용
                st.info("ℹ️ rembg 라이브러리가 없어 원본 이미지로 분석합니다. (`pip install rembg`로 설치 가능)")
            
            # 이미지 표시 (원본/배경제거 비교)
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="원본 이미지", width='stretch')
            with col_img2:
                if bg_removed:
                    st.image(processed_image, caption="배경 제거 이미지 ✅", width='stretch')
                    st.success("배경 제거 성공!")
                else:
                    st.image(processed_image, caption="처리된 이미지 (원본 사용)", width='stretch')
                    if REMBG_AVAILABLE and bg_error:
                        with st.expander("🔍 배경 제거 오류 상세"):
                            st.error(bg_error)
                            st.info("""
                            **해결 방법:**
                            1. rembg 재설치: `pip uninstall rembg && pip install rembg`
                            2. 모델 다운로드 확인: 첫 실행 시 모델이 자동 다운로드됩니다
                            3. 인터넷 연결 확인: 모델 다운로드에 인터넷이 필요합니다
                            """)
                    elif REMBG_AVAILABLE:
                        st.warning("⚠️ 배경 제거가 완전히 수행되지 않았습니다.")
            
            # 얼굴 및 체형 분석
            st.subheader("👤 얼굴 및 체형 분석")
            with st.spinner("얼굴 및 체형 분석 중..."):
                face_info = st.session_state.body_analyzer.analyze_face(processed_image)
                body_info = st.session_state.body_analyzer.analyze_body(processed_image)
                
                # 성별 자동 인식 (이미지가 변경된 경우에만)
                import hashlib
                current_image_hash = hashlib.md5(processed_image.tobytes()).hexdigest()
                
                # last_image_hash 초기화 확인
                if 'last_image_hash' not in st.session_state:
                    st.session_state.last_image_hash = None
                
                # 이미지 해시 저장 (성별 인식은 result 생성 후 수행)
                if current_image_hash != st.session_state.last_image_hash:
                    st.session_state.last_image_hash = current_image_hash
            
            # 분석 결과 표시
            col_face, col_body = st.columns(2)
            with col_face:
                if face_info.get("detected"):
                    st.success("✅ 얼굴 탐지됨")
                    st.write(f"**얼굴 형태:** {face_info.get('face_shape', '알 수 없음')}")
                    st.write(f"**눈 크기:** {face_info.get('eye_size', '알 수 없음')}")
                    if face_info.get("face_ratio"):
                        st.caption(f"얼굴 비율: {face_info.get('face_ratio', 0):.2f}")
                    
                    # DeepFace 분석 결과 표시
                    if face_info.get("age"):
                        st.write(f"**추정 나이:** {face_info.get('age')}세")
                    if face_info.get("emotion"):
                        emotion_map = {
                            "happy": "😊 행복",
                            "sad": "😢 슬픔",
                            "angry": "😠 화남",
                            "surprise": "😮 놀람",
                            "fear": "😨 두려움",
                            "disgust": "🤢 혐오",
                            "neutral": "😐 무표정"
                        }
                        emotion = face_info.get("emotion", "")
                        emotion_display = emotion_map.get(emotion, emotion)
                        st.write(f"**감정:** {emotion_display}")
                    if face_info.get("gender_deepface"):
                        st.write(f"**DeepFace 성별 인식:** {face_info.get('gender_deepface')}")
                else:
                    st.warning("⚠️ 얼굴을 찾을 수 없습니다")
                    message = face_info.get("message", "얼굴이 명확하게 보이도록 이미지를 업로드해주세요.")
                    st.info(message)
                    if face_info.get("hint"):
                        st.caption(f"💡 {face_info.get('hint')}")
            
            with col_body:
                if body_info.get("detected"):
                    st.success("✅ 체형 분석됨")
                    st.write(f"**체형:** {body_info.get('body_type', '알 수 없음')}")
                    if body_info.get("body_ratio"):
                        st.write(f"**체형 비율:** {body_info.get('body_ratio', 0):.2f}")
                else:
                    st.warning("⚠️ 체형을 분석할 수 없습니다")
                    st.info(body_info.get("message", "전신 사진을 업로드해주세요."))
            
            # 코디 추천 결과 표시 (배경 제거 이미지 사용, 얼굴/체형 정보 포함)
            # 먼저 YOLO/CLIP 분석 실행 (점수 계산을 위해)
            fr = st.session_state.fashion_recommender
            result = fr.recommend_outfit(processed_image, mbti_type, temperature, weather, season)
            
            # 성별 자동 인식 (얼굴 특징 기반 + DeepFace + 의상 기반 + CLIP)
            gender_changed = False
            if current_image_hash != st.session_state.get('last_gender_hash', None):
                # 방법 1: 얼굴 특징 기반 성별 인식 (MediaPipe 얼굴 분석 결과 활용)
                # 이미 analyze_face가 호출되어 face_info에 결과가 있음
                detected_gender = None
                
                # 얼굴 특징 기반 추정 시도
                if face_info and face_info.get("detected"):
                    detected_gender = st.session_state.body_analyzer._estimate_gender_from_features(face_info)
                
                # 방법 2: DeepFace 사용 (설치된 경우)
                if not detected_gender:
                    detected_gender = st.session_state.body_analyzer.detect_gender(processed_image)
                
                # 방법 3: 의상 기반 판단
                if not detected_gender:
                    detected_gender = detect_gender_from_image(
                        processed_image, 
                        fr.analyzer,
                        result
                    )
                
                if detected_gender and detected_gender != "공용":
                    # 기존 성별과 비교하여 변경 여부 확인
                    old_gender = st.session_state.get('auto_gender')
                    st.session_state.auto_gender = detected_gender
                    gender_index_map = {"남성": 0, "여성": 1, "공용": 2}
                    new_gender_index = gender_index_map.get(detected_gender, 0)
                    
                    # 성별이 변경되었거나 처음 인식하는 경우
                    if old_gender != detected_gender or st.session_state.selected_gender != new_gender_index:
                        st.session_state.selected_gender = new_gender_index
                        st.session_state.gender_auto_update_pending = True  # rerun 후 업데이트 플래그
                        gender_changed = True
                
                st.session_state.last_gender_hash = current_image_hash
                
                # 성별이 변경되었으면 즉시 사이드바 반영
                if gender_changed:
                    st.rerun()
            
            # 외모 및 패션 점수 계산 (향상된 시스템 사용)
            appearance_scores = st.session_state.scoring_system.score_appearance(
                face_info, body_info, image=processed_image
            )
            fashion_scores = st.session_state.scoring_system.score_fashion(
                result.get("detected_items", {}).get("items", []),
                result.get("style_analysis", {}),
                weather,
                season,
                temperature,
                image=processed_image  # 이미지 전달 (향상된 분석용)
            )
            
            # 점수 표시
            st.subheader("📊 외모 및 패션 점수")
            
            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.markdown("### 👤 외모 점수")
                st.metric("얼굴", f"{appearance_scores['얼굴']}/100", 
                         delta=f"{appearance_scores['얼굴'] - 70}", 
                         delta_color="normal" if appearance_scores['얼굴'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(appearance_scores['얼굴']))
                
                st.metric("체형", f"{appearance_scores['체형']}/100",
                         delta=f"{appearance_scores['체형'] - 70}",
                         delta_color="normal" if appearance_scores['체형'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(appearance_scores['체형']))
                
                st.metric("전체 외모", f"{appearance_scores['전체 외모']}/100",
                         delta=f"{appearance_scores['전체 외모'] - 70}",
                         delta_color="normal" if appearance_scores['전체 외모'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(appearance_scores['전체 외모']))
            
            with col_score2:
                st.markdown("### 👗 패션 점수")
                st.metric("아이템 구성", f"{fashion_scores['아이템 구성']}/100",
                         delta=f"{fashion_scores['아이템 구성'] - 70}",
                         delta_color="normal" if fashion_scores['아이템 구성'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(fashion_scores['아이템 구성']))
                
                st.metric("스타일 일치도", f"{fashion_scores['스타일 일치도']}/100",
                         delta=f"{fashion_scores['스타일 일치도'] - 70}",
                         delta_color="normal" if fashion_scores['스타일 일치도'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(fashion_scores['스타일 일치도']))
                
                st.metric("계절 적합성", f"{fashion_scores['계절 적합성']}/100",
                         delta=f"{fashion_scores['계절 적합성'] - 70}",
                         delta_color="normal" if fashion_scores['계절 적합성'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(fashion_scores['계절 적합성']))
                
                st.metric("날씨 적합성", f"{fashion_scores['날씨 적합성']}/100",
                         delta=f"{fashion_scores['날씨 적합성'] - 70}",
                         delta_color="normal" if fashion_scores['날씨 적합성'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(fashion_scores['날씨 적합성']))
                
                st.metric("전체 패션", f"{fashion_scores['전체 패션']}/100",
                         delta=f"{fashion_scores['전체 패션'] - 70}",
                         delta_color="normal" if fashion_scores['전체 패션'] >= 70 else "inverse")
                st.caption(st.session_state.scoring_system.get_score_label(fashion_scores['전체 패션']))
            
            # 상세 피드백
            feedback = st.session_state.scoring_system.get_detailed_feedback(appearance_scores, fashion_scores, season)
            if feedback:
                with st.expander("💡 개선 제안"):
                    for fb in feedback:
                        st.write(fb)
            
            # 코디 추천 결과 표시
            display_outfit_recommendations(
                processed_image, mbti_type, temperature, weather, season, 
                gender, debug_mode, face_info, body_info, original_image=image,
                precomputed_result=result, appearance_scores=appearance_scores, fashion_scores=fashion_scores
            )
    
    with tab2:
        # 텍스트 기반 코디 검색
        st.subheader("🔍 텍스트 기반 코디 검색")
        
        # 세션 상태 초기화
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""
        
        # 빠른 선택 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎉 파티용 코디"):
                st.session_state.search_query = "파티용 코디"
        with col2:
            if st.button("💼 출근룩"):
                st.session_state.search_query = "출근룩"
        with col3:
            if st.button("💕 데이트룩"):
                st.session_state.search_query = "데이트룩"
        
        search_query = st.text_input(
            "원하는 코디를 검색하세요", 
            value=st.session_state.search_query,
            placeholder="예: 파티용 코디, 출근룩, 데이트룩"
        )
        
        if search_query:
            st.session_state.search_query = search_query
            display_text_search_results(search_query, mbti_type)
    
    with tab3:
        # 트렌드 및 인기 코디
        st.subheader("🔥 이번 시즌 인기 코디")
        display_trend_outfits(season)
    
    with tab4:
        # 모델 관리 페이지
        display_model_manager()

def display_outfit_recommendations(image, mbti, temp, weather, season, gender, debug_mode=False, 
                                   face_info=None, body_info=None, original_image=None,
                                   precomputed_result=None, appearance_scores=None, fashion_scores=None):
    """코디 추천 결과 표시"""
    # 통합 추천 + 탐지/분석 실행 (이미 계산된 경우 재사용)
    if precomputed_result is None:
        fr = st.session_state.fashion_recommender
        result = fr.recommend_outfit(image, mbti, temp, weather, season)
    else:
        result = precomputed_result
    
    # 이미지 분석 결과를 추천에 반영
    detected_items_data = result.get("detected_items", {})
    style_analysis_data = result.get("style_analysis", {})
    
    recommendations = st.session_state.recommendation_engine.get_personalized_recommendation(
        mbti, temp, weather, season,
        detected_items=detected_items_data.get("items", []),
        style_analysis=style_analysis_data
    )

    # 진단 모드: YOLO/CLIP 상세 출력
    if debug_mode:
        with st.expander("🧪 모델 진단 (YOLO/CLIP)", expanded=True):
            det = result.get("detected_items", {}).get("items", [])
            vis_img = draw_detections(image, det) if det else image
            st.image(vis_img, caption="YOLO 탐지 시각화", width='stretch')

            # 탐지 표
            if det:
                st.markdown("**YOLO 탐지 결과**")
                img_w, img_h = image.size
                st.info(f"📐 이미지 크기: {img_w} x {img_h} 픽셀")
                
                for i, d in enumerate(det, 1):
                    bbox = d.get('bbox', [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        width = x2 - x1
                        height = y2 - y1
                        area_ratio = (width * height) / (img_w * img_h) * 100 if (img_w * img_h) > 0 else 0
                        
                        class_display = d.get('class', '?')
                        original_class = d.get('original_class', '')
                        class_en = d.get('class_en', '')
                        
                        # CLIP 검증으로 수정된 경우 표시
                        if original_class and original_class != class_en:
                            st.write(f"{i}. **{class_display}** (신뢰도: {d.get('confidence',0):.2f})")
                            st.caption(f"   🔄 YOLO 원본: {original_class} → CLIP 검증 후: {class_display}")
                            st.success("✅ CLIP 검증으로 정정되었습니다")
                        else:
                            st.write(f"{i}. **{class_display}** (신뢰도: {d.get('confidence',0):.2f})")
                        
                        st.write(f"   - 바운딩박스: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                        st.write(f"   - 크기: {width:.0f} x {height:.0f} (이미지의 {area_ratio:.1f}%)")
                        
                        # COCO 모델 경고
                        if d.get('class') == 'person':
                            st.warning("⚠️ COCO 모델은 'person'만 탐지합니다. 패션 아이템 세부 탐지는 패션 전용 모델 학습이 필요합니다.")
                    else:
                        st.write(f"{i}. {d.get('class','?')} (conf {d.get('confidence',0):.2f}) bbox=잘못된 형식")
            else:
                st.info("탐지된 아이템이 없습니다.")

            # CLIP 유사도 상위 K
            sa = result.get("style_analysis", {})
            matches = sa.get("text_matches", {})
            if matches:
                st.markdown("**CLIP 유사도 상위 항목**")
                st.info(f"📊 분석된 키워드 수: {len(matches)}개")
                
                # 색상과 스타일 분리
                color_keywords = ['색', 'color', 'red', 'blue', 'white', 'black', 'yellow', 'green', 'purple', 'pink', 'orange', 'navy', 'khaki', 'beige', 'gray', 'grey']
                color_matches = {k: matches[k] for k in matches.keys() if any(c in k.lower() for c in color_keywords)}
                style_matches = {k: matches[k] for k in matches.keys() if k not in color_matches}
                
                if color_matches:
                    st.markdown("**🎨 색상 유사도**")
                    top_colors = sorted(color_matches.items(), key=lambda x: x[1], reverse=True)[:10]
                    for k, v in top_colors:
                        st.write(f"- {k}: {v:.3f}")
                
                if style_matches:
                    st.markdown("**👔 스타일 유사도**")
                    top_styles = sorted(style_matches.items(), key=lambda x: x[1], reverse=True)[:10]
                    for k, v in top_styles:
                        st.write(f"- {k}: {v:.3f}")
                
                # 전체 상위 10개
                top = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:10]
                try:
                    import pandas as pd
                    import altair as alt
                    df = pd.DataFrame(top, columns=["label","score"])
                    chart = alt.Chart(df).mark_bar().encode(x='label', y='score')
                    st.altair_chart(chart, use_container_width=False)
                except Exception:
                    pass
            else:
                st.info("CLIP 유사도 결과가 없습니다.")

            # 원시 결과 미리보기
            import json
            st.markdown("**원시 결과 미리보기**")
            preview = {
                "detected_items": result.get("detected_items", {}).get("items", []),
                "style_analysis": {
                    k: v for k, v in sa.items() if k in ("style","color","confidence")
                }
            }
            st.code(json.dumps(preview, ensure_ascii=False, indent=2), language="json")
    
    st.subheader("🎯 추천 코디 (3가지 버전)")
    
    # 이미지 분석 결과 기반 동적 스타일 선택
    image_suggestions = recommendations.get("image_suggestions", {})
    style_matches = image_suggestions.get("style_matches", {})
    image_based_combinations = image_suggestions.get("recommendation_based_on_image", [])
    
    # CLIP 스타일 점수 기반으로 스타일 순서 결정
    if style_matches:
        # 점수가 높은 순으로 정렬
        sorted_styles = sorted(style_matches.items(), key=lambda x: x[1], reverse=True)
        top_styles = [style[0] for style in sorted_styles[:3]]
        
        # 기본 스타일과 결합 (고정된 3개가 아닌 동적 선택)
        outfit_styles_list = []
        for style in ["캐주얼", "포멀", "트렌디"]:
            if style in top_styles:
                outfit_styles_list.append(style)
        
        # 부족하면 기본 스타일로 채움
        for style in ["캐주얼", "포멀", "트렌디"]:
            if len(outfit_styles_list) < 3 and style not in outfit_styles_list:
                outfit_styles_list.append(style)
        
        outfit_styles = outfit_styles_list[:3]
    else:
        outfit_styles = ["캐주얼", "포멀", "트렌디"]
    
    # 이미지 기반 조합이 있으면 우선 사용
    has_image_based = len(image_based_combinations) > 0
    
    # 3가지 버전 코디 추천
    col1, col2, col3 = st.columns(3)
    
    # 각 버전별 설명 생성 (이미지 분석 결과 반영)
    # 색상 추천 추출 (CLIP 분석 결과 활용)
    color_suggestions = image_suggestions.get("color_matches", {})
    top_colors = []
    if color_suggestions:
        top_colors = sorted(color_suggestions.items(), key=lambda x: x[1], reverse=True)[:3]
    
    outfit_descriptions = []
    for idx, style in enumerate(outfit_styles):
        if has_image_based and idx < len(image_based_combinations):
            # 이미지 기반 조합 우선 사용
            combo = image_based_combinations[idx]
            reason = combo.get("reason", f"{style} 스타일")
            # 색상 추천 추가
            if top_colors and idx < len(top_colors):
                color_name = top_colors[idx][0]
                reason += f", {color_name} 톤 추천"
            outfit_descriptions.append(reason)
        else:
            # 기존 방식 (MBTI/계절/날씨 기반) + 색상 추천
            base_desc = ""
            if idx == 0:
                base_desc = f"{recommendations['mbti_style']['style']} 스타일"
                # MBTI 색상 추가
                if recommendations['mbti_style'].get('colors'):
                    base_desc += f", {recommendations['mbti_style']['colors'][0]} 톤"
            elif idx == 1:
                base_desc = f"{recommendations['seasonal_info']['mood']}한 {recommendations['seasonal_info']['materials'][0]} 소재"
                # 계절 색상 추가
                if recommendations['seasonal_info'].get('colors'):
                    base_desc += f", {recommendations['seasonal_info']['colors'][0]} 톤"
            else:
                base_desc = f"{recommendations['weather_info']['mood']}한 스타일"
                # 이미지 분석 색상 추가 (있는 경우)
                if top_colors:
                    base_desc += f", {top_colors[0][0]} 톤 추천"
            outfit_descriptions.append(base_desc)
    
    for idx, (col, style, desc) in enumerate(zip([col1, col2, col3], outfit_styles, outfit_descriptions)):
        with col:
            st.write(f"**추천 코디 {idx+1}**")
            st.write(f"**{style} 스타일**")
            
            # CLIP 점수 표시 (있는 경우)
            if style_matches and style in style_matches:
                score = style_matches[style]
                st.caption(f"📊 이미지 분석 점수: {score:.2f}")
            
            st.info(desc)
            st.write(f"**아이템:**")
            
            # 표시될 아이템 텍스트 수집 (이미지 생성에 사용)
            displayed_items = []
            
            # 이미지 기반 조합이 있으면 사용
            if has_image_based and idx < len(image_based_combinations):
                combo = image_based_combinations[idx]
                items = combo.get("items", [])
                for item in items:
                    displayed_items.append(item)
                    st.write(f"• {item}")
            else:
                # 기존 방식 (템플릿 기반)
                if idx == 0:
                    # 이미지 색상 우선 사용, 없으면 MBTI 색상
                    detected_colors = image_suggestions.get("color_matches", {})
                    if detected_colors:
                        top_color = max(detected_colors.items(), key=lambda x: x[1])[0]
                        color_display = top_color
                    else:
                        color_display = recommendations['mbti_style']['colors'][0]
                    
                    item1 = f"{color_display} 상의"
                    item2 = f"{recommendations['seasonal_info']['colors'][0]} 하의"
                    displayed_items = [item1, item2]
                    st.write(f"• {item1}")
                    st.write(f"• {item2}")
                elif idx == 1:
                    item1 = f"{recommendations['seasonal_info']['materials'][0]} 재킷"
                    item2 = f"{recommendations['seasonal_info']['colors'][0]} 바지"
                    displayed_items = [item1, item2]
                    st.write(f"• {item1}")
                    st.write(f"• {item2}")
                else:
                    item1 = recommendations['weather_info']['accessories'][0]
                    item2 = f"{recommendations['temperature_guidance']['material']} 재킷"
                    displayed_items = [item1, item2]
                    st.write(f"• {item1}")
                    st.write(f"• {item2}")
            
            # 구체 제품 추천
            products = st.session_state.recommendation_engine.recommend_products(style, gender)
            st.write("**추천 제품:**")
            for p in products:
                st.write(f"• {p}")
            
            # AI 생성 이미지 (자동 생성 또는 버튼)
            if 'enable_ai_images' in st.session_state and st.session_state.enable_ai_images:
                try:
                    from src.utils.image_generator import OutfitImageGenerator
                    
                    # 이미지 생성기 초기화 (세션 상태나 API 키 변경 시 재초기화)
                    current_method = st.session_state.get("image_gen_method", "huggingface_api")
                    current_hf_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
                    
                    # 재초기화가 필요한 경우 확인
                    need_reinit = (
                        'image_generator' not in st.session_state or
                        st.session_state.get('last_image_gen_method') != current_method or
                        (current_method == "huggingface_api" and 
                         st.session_state.get('last_hf_api_key') != current_hf_key)
                    )
                    
                    if need_reinit:
                        # 프로토타입 사용 설정 (Stable Diffusion 로컬만)
                        use_prototype = current_method == "stable_diffusion"
                        st.session_state.image_generator = OutfitImageGenerator(
                            method=current_method,
                            use_prototype=use_prototype
                        )
                        st.session_state.last_image_gen_method = current_method
                        if current_method == "huggingface_api":
                            st.session_state.last_hf_api_key = current_hf_key
                    
                    # 코디 설명 구성 - 표시된 아이템 텍스트를 그대로 사용
                    outfit_desc = {
                        "items": displayed_items,  # ✅ 표시된 텍스트 그대로 사용
                        "style": style,
                        "colors": [color_display] if idx == 0 and 'color_display' in locals() else recommendations.get('seasonal_info', {}).get('colors', [])[:2],
                        "gender": gender  # 성별 정보 추가
                    }
                    
                    # 자동 생성 여부 확인
                    auto_generate = st.session_state.get("auto_generate_images", False)
                    num_auto_images = st.session_state.get("num_auto_images", 1)
                    should_auto_generate = auto_generate and idx < num_auto_images
                    
                    # 이미지 생성 캐시 키 (이미지 해시 + 스타일 + 인덱스로 고유하게)
                    current_image_hash = st.session_state.get("last_image_hash", "default")
                    cache_key = f"generated_image_{current_image_hash}_{style}_{idx}"
                    
                    # 자동 생성 또는 캐시된 이미지 사용
                    if should_auto_generate:
                        if cache_key not in st.session_state:
                            with st.spinner(f"🎨 {style} 스타일 AI 이미지 생성 중... (10-30초 소요)"):
                                generated_image = st.session_state.image_generator.generate_outfit_image(
                                    outfit_desc, style_info=recommendations
                                )
                                if generated_image:
                                    st.session_state[cache_key] = generated_image
                                    st.image(generated_image, caption=f"{style} 스타일 AI 생성 이미지", width='stretch')
                                    st.success("✅ 이미지 생성 완료")
                                else:
                                    st.warning("⚠️ 이미지 생성 실패")
                                    with st.expander("🔍 문제 해결 가이드", expanded=True):
                                        st.markdown("""
                                        ### ⚠️ **현재 상황: Hugging Face API 제한**
                                        
                                        Hugging Face의 정책 변경으로 무료 계정에서 Inference API 사용이 제한되었습니다.
                                        Read 토큰으로도 403/404 오류가 계속 발생한다면 **다른 방법 사용을 권장**합니다.
                                        
                                        ---
                                        
                                        ### 💡 **추천 해결 방법 (우선순위 순)**
                                        
                                        #### **방법 1: DALL-E API 사용** ⭐ 가장 안정적
                                        
                                        1. OpenAI 계정 생성: https://platform.openai.com
                                        2. API 키 발급 (결제 정보 필요)
                                        3. 사이드바 → "이미지 생성 방법" → **"dall_e"** 선택
                                        4. 환경 변수 설정:
                                           ```bash
                                           export OPENAI_API_KEY="your-api-key"
                                           ```
                                        💰 비용: $0.04/image (1024x1024)
                                        
                                        #### **방법 2: Stable Diffusion 로컬 실행** ⭐ M2 맥북 최적화 (무료)
                                        
                                        1. 라이브러리 설치:
                                           ```bash
                                           pip install diffusers accelerate
                                           ```
                                        2. 사이드바 → "이미지 생성 방법" → **"stable_diffusion"** 선택
                                        3. 자동으로 Apple Silicon (M2) 감지 및 최적화
                                        
                                        **특징:**
                                        - ✅ 완전 무료 (API 비용 없음)
                                        - ✅ M2 맥북 최적화 (MPS 백엔드 자동 사용)
                                        - ✅ 오프라인 작동 가능
                                        - ⏱️ 생성 시간: 약 30-60초 (M2 기준)
                                        - 💾 첫 실행 시 모델 다운로드 (약 4GB, 한 번만)
                                        
                                        #### **방법 3: 이미지 생성 비활성화**
                                        
                                        - 사이드바 → "AI 이미지 생성 활성화" → **OFF**
                                        - 텍스트 기반 추천만 사용
                                        
                                        ---
                                        
                                        ### ❌ **계속 시도해도 안 되는 경우**
                                        
                                        - Hugging Face Pro 계정 업그레이드 (유료, $9/month)
                                        - 또는 위 대안 방법 사용 권장
                                        """)
                        else:
                            # 캐시된 이미지 사용
                            cached_image = st.session_state[cache_key]
                            st.image(cached_image, caption=f"{style} 스타일 AI 생성 이미지", width='stretch')
                            st.success("✅ 이미지 생성 완료 (캐시)")
                    else:
                        # 수동 생성 버튼
                        gen_button_key = f"generate_image_{idx}"
                        if st.button(f"🎨 {style} 스타일 이미지 생성", key=gen_button_key):
                            with st.spinner(f"AI 이미지 생성 중... (10-30초 소요)"):
                                generated_image = st.session_state.image_generator.generate_outfit_image(
                                    outfit_desc, style_info=recommendations
                                )
                                if generated_image:
                                    st.session_state[cache_key] = generated_image
                                    st.image(generated_image, caption=f"{style} 스타일 AI 생성 이미지", width='stretch')
                                    st.success("✅ 이미지 생성 완료")
                                else:
                                    st.warning("⚠️ 이미지 생성 실패")
                                    with st.expander("🔍 문제 해결 가이드", expanded=True):
                                        st.markdown("""
                                        ### ⚠️ **현재 상황: Hugging Face API 제한**
                                        
                                        Hugging Face의 정책 변경으로 무료 계정에서 Inference API 사용이 제한되었습니다.
                                        Read 토큰으로도 403/404 오류가 계속 발생한다면 **다른 방법 사용을 권장**합니다.
                                        
                                        ---
                                        
                                        ### 💡 **추천 해결 방법 (우선순위 순)**
                                        
                                        #### **방법 1: DALL-E API 사용** ⭐ 가장 안정적
                                        
                                        1. OpenAI 계정 생성: https://platform.openai.com
                                        2. API 키 발급 (결제 정보 필요)
                                        3. 사이드바 → "이미지 생성 방법" → **"dall_e"** 선택
                                        4. 환경 변수 설정:
                                           ```bash
                                           export OPENAI_API_KEY="your-api-key"
                                           ```
                                        💰 비용: $0.04/image (1024x1024)
                                        
                                        #### **방법 2: Stable Diffusion 로컬 실행** ⭐ M2 맥북 최적화 (무료)
                                        
                                        1. 라이브러리 설치:
                                           ```bash
                                           pip install diffusers accelerate
                                           ```
                                        2. 사이드바 → "이미지 생성 방법" → **"stable_diffusion"** 선택
                                        3. 자동으로 Apple Silicon (M2) 감지 및 최적화
                                        
                                        **특징:**
                                        - ✅ 완전 무료 (API 비용 없음)
                                        - ✅ M2 맥북 최적화 (MPS 백엔드 자동 사용)
                                        - ✅ 오프라인 작동 가능
                                        - ⏱️ 생성 시간: 약 30-60초 (M2 기준)
                                        - 💾 첫 실행 시 모델 다운로드 (약 4GB, 한 번만)
                                        
                                        #### **방법 3: 이미지 생성 비활성화**
                                        
                                        - 사이드바 → "AI 이미지 생성 활성화" → **OFF**
                                        - 텍스트 기반 추천만 사용
                                        
                                        ---
                                        
                                        ### ❌ **계속 시도해도 안 되는 경우**
                                        
                                        - Hugging Face Pro 계정 업그레이드 (유료, $9/month)
                                        - 또는 위 대안 방법 사용 권장
                                        """)
                except ImportError:
                    st.caption("💡 AI 이미지 생성을 사용하려면 `pip install diffusers` 또는 API 키 설정이 필요합니다.")
                except Exception as e:
                    st.caption(f"💡 이미지 생성 기능 준비 중: {str(e)[:50]}")
            
            # 탐지된 아이템과 조화로운 아이템 표시
            if image_suggestions and image_suggestions.get("detected_items_info"):
                detected_info = image_suggestions["detected_items_info"]
                if detected_info and idx == 0:  # 첫 번째 버전에만 표시
                    item = detected_info[0]
                    complementary = item.get("complementary_items", [])
                    if complementary:
                        st.caption(f"💡 현재 {item['item']}와 조화: {', '.join(complementary[:2])}")
    
    # 이미지 기반 추천 상세 정보 (있는 경우)
    if image_suggestions and (image_suggestions.get("detected_items_info") or image_suggestions.get("style_matches")):
        with st.expander("🖼️ 이미지 분석 기반 추천 상세", expanded=False):
            if image_suggestions.get("detected_items_info"):
                st.markdown("**탐지된 아이템:**")
                for item_info in image_suggestions["detected_items_info"][:3]:
                    item_name = item_info.get("item", "")
                    confidence = item_info.get("confidence", 0)
                    complementary = item_info.get("complementary_items", [])
                    st.write(f"• **{item_name}** (신뢰도: {confidence:.2f})")
                    if complementary:
                        st.caption(f"  → 조화로운 아이템: {', '.join(complementary)}")
            
            if image_suggestions.get("style_matches"):
                st.markdown("**CLIP 스타일 분석:**")
                sorted_styles = sorted(image_suggestions["style_matches"].items(), 
                                     key=lambda x: x[1], reverse=True)
                for style_name, score in sorted_styles[:5]:
                    st.write(f"• {style_name}: {score:.3f}")
            
            if image_suggestions.get("color_matches"):
                st.markdown("**CLIP 색상 분석:**")
                sorted_colors = sorted(image_suggestions["color_matches"].items(), 
                                     key=lambda x: x[1], reverse=True)
                for color_name, score in sorted_colors[:5]:
                    st.write(f"• {color_name}: {score:.3f}")
    
    # 추천 이유
    st.subheader("💡 이 조합이 어울리는 이유")
    for reason in recommendations['recommendation_reason']:
        st.write(reason)
    
    # 롤모델 및 화장법
    st.subheader("🌟 롤모델 스타일 참고")
    for style in outfit_styles:
        celebrity = st.session_state.recommendation_engine.get_celebrity_style_reference(style)
        st.write(f"**{style} 스타일:** {celebrity}")
    
    st.subheader("💄 추천 화장법")
    for style in outfit_styles:
        makeup = st.session_state.recommendation_engine.get_makeup_suggestions(style, mbti)
        st.write(f"**{style} 스타일:** {makeup}")

    # 얼굴/체형 기반 개인화 추천
    if face_info and body_info:
        body_recommendations = st.session_state.body_analyzer.get_recommendation_based_on_body(
            face_info if face_info else {},
            body_info if body_info else {}
        )
        if body_recommendations:
            st.subheader("👤 체형 맞춤 추천")
            for rec in body_recommendations:
                st.info(f"💡 {rec}")
    
    # 현재 코디 평가
    st.subheader("🧭 현재 코디 평가")
    eval_result = st.session_state.recommendation_engine.evaluate_current_outfit(
        result.get("detected_items", {}).get("items", []),
        result.get("style_analysis", {}),
        weather,
        season
    )
    st.write(f"**점수:** {eval_result['score']} / 100 ({eval_result['label']})")
    st.write("**피드백:**")
    for fb in eval_result["feedback"]:
        st.write(f"• {fb}")
    
    # 얼굴/체형 정보 추가 피드백
    if face_info and face_info.get("detected"):
        st.write(f"• 얼굴 형태({face_info.get('face_shape')})에 맞는 넥라인 추천")
    if body_info and body_info.get("detected"):
        st.write(f"• 체형({body_info.get('body_type')})에 최적화된 실루엣 추천")

def display_text_search_results(query, mbti):
    """텍스트 검색 결과 표시"""
    results = st.session_state.recommendation_engine.search_text_based_outfits(query)
    
    st.subheader(f"'{query}' 검색 결과")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**카테고리:** {results['category']}")
        st.write(f"**무드:** {results['mood']}")
        st.write(f"**추천 색상:** {', '.join(results['colors'])}")
    
    with col2:
        st.write("**추천 아이템:**")
        for item in results['items']:
            st.write(f"• {item}")
    
    # MBTI 개인화 적용
    if mbti in MBTI_STYLES:
        st.info(f"💡 {mbti} 유형을 위해 {MBTI_STYLES[mbti]['style']} 요소가 추가로 반영되었습니다.")
    
    # 롤모델 및 화장법
    st.subheader("🌟 관련 롤모델")
    celebrity = st.session_state.recommendation_engine.get_celebrity_style_reference(results['category'])
    st.write(celebrity)
    
    st.subheader("💄 추천 화장법")
    makeup = st.session_state.recommendation_engine.get_makeup_suggestions(results['category'], mbti)
    st.write(makeup)

def display_trend_outfits(season):
    """트렌드 코디 표시"""
    # SNS 트렌드 분석 결과 (실제 SNS 크롤링은 향후 구현 예정)
    trend_outfits = {
        "봄": {
            "trends": ["파스텔 톤 코디", "플라워 프린트", "라이트 재킷"],
            "colors": ["라벤더", "피치", "민트"],
            "description": "이번 봄 트렌드는 파스텔 톤과 플라워 프린트입니다!"
        },
        "여름": {
            "trends": ["미니멀 화이트", "린넨 코디", "비치웨어 스타일"],
            "colors": ["화이트", "베이지", "아쿠아"],
            "description": "시원한 여름을 위한 미니멀 화이트 코디가 인기입니다!"
        },
        "가을": {
            "trends": ["어스톤 코디", "오버사이즈 코트", "니트 레이어링"],
            "colors": ["터키석", "머스타드", "버건디"],
            "description": "따뜻한 가을을 위한 어스톤 톤이 유행 중입니다!"
        },
        "겨울": {
            "trends": ["다크 레더", "플리스 코디", "패딩 스타일"],
            "colors": ["블랙", "네이비", "그레이"],
            "description": "우아한 겨울을 위한 다크 톤 코디가 트렌드입니다!"
        }
    }
    
    trend = trend_outfits.get(season, trend_outfits["봄"])
    
    st.info(trend['description'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**인기 트렌드 스타일:**")
        for trend_item in trend['trends']:
            st.write(f"• {trend_item}")
    
    with col2:
        st.write("**인기 컬러:**")
        for color in trend['colors']:
            st.write(f"• {color}")
    
    st.subheader("🔥 이번 시즌 Top 3 코디")
    
    for i, trend_item in enumerate(trend['trends'][:3], 1):
        with st.expander(f"코디 {i}: {trend_item}"):
            st.write(f"**스타일:** {trend_item}")
            st.write(f"**추천 컬러:** {trend['colors'][i-1] if i <= len(trend['colors']) else trend['colors'][0]}")
            st.write(f"**계절:** {season}")
            celebrity = st.session_state.recommendation_engine.get_celebrity_style_reference("트렌디")
            st.write(f"**참고 스타일:** {celebrity}")

def display_model_manager():
    """모델 관리자 페이지"""
    st.title("⚙️ 모델 관리자")
    st.markdown("YOLOv5와 CLIP 모델의 상태를 확인하고 관리합니다.")
    
    # 서브탭 구성
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 모델 상태", 
        "💻 시스템 정보", 
        "🎓 학습 관리",
        "🔧 유틸리티"
    ])
    
    with sub_tab1:
        st.subheader("📊 모델 상태")
        
        col1, col2 = st.columns(2)
        
        # YOLOv5 상태
        with col1:
            st.markdown("### 🎯 YOLOv5 모델")
            yolo_status = st.session_state.model_manager.get_yolo_status(
                st.session_state.fashion_recommender.detector
            )
            
            if yolo_status["loaded"]:
                st.success("✅ 모델 로드됨")
                st.write(f"**모델:** {yolo_status['model_name']}")
                if yolo_status["model_path"]:
                    st.write(f"**경로:** {yolo_status['model_path']}")
                if yolo_status["model_size"]:
                    st.write(f"**크기:** {yolo_status['model_size']}")
            else:
                st.warning("⚠️ 모델이 로드되지 않음")
            
            if yolo_status["error"]:
                st.error(f"오류: {yolo_status['error']}")
            
            st.markdown("#### 사용 가능한 모델")
            for model in yolo_status["available_models"][:5]:
                st.write(f"• {model}")
            if len(yolo_status["available_models"]) > 5:
                st.write(f"... 총 {len(yolo_status['available_models'])}개")
        
        # CLIP 상태
        with col2:
            st.markdown("### 🖼️ CLIP 모델")
            clip_status = st.session_state.model_manager.get_clip_status(
                st.session_state.fashion_recommender.analyzer
            )
            
            if clip_status["loaded"]:
                st.success("✅ 모델 로드됨")
                st.write(f"**모델:** {clip_status['model_name']}")
                st.write(f"**장치:** {clip_status['device']} ({clip_status['device_type']})")
                
                if clip_status["config"]:
                    st.write(f"**파라미터 수:** {clip_status['config']['total_parameters']}")
                
                if clip_status["memory_usage"]:
                    st.write(f"**GPU 메모리 사용:** {clip_status['memory_usage']['allocated_gb']} GB")
                    st.write(f"**예약된 메모리:** {clip_status['memory_usage']['reserved_gb']} GB")
            else:
                st.warning("⚠️ 모델이 로드되지 않음")
            
            if clip_status["error"]:
                st.error(f"오류: {clip_status['error']}")
        
        # 새로고침 버튼
        if st.button("🔄 상태 새로고침"):
            st.rerun()
    
    with sub_tab2:
        st.subheader("💻 시스템 정보")
        system_info = st.session_state.model_manager.get_system_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 소프트웨어")
            st.write(f"**Python 버전:** {system_info['python_version']}")
            st.write(f"**PyTorch 버전:** {system_info['pytorch_version']}")
            st.write(f"**CUDA 사용 가능:** {'✅ 예' if system_info['cuda_available'] else '❌ 아니오'}")
            if system_info["cuda_version"]:
                st.write(f"**CUDA 버전:** {system_info['cuda_version']}")
            if system_info["gpu_name"]:
                st.write(f"**GPU:** {system_info['gpu_name']}")
        
        with col2:
            st.markdown("### 💾 하드웨어")
            st.write(f"**CPU 코어 수:** {system_info['cpu_count']}")
            st.write(f"**메모리 총량:** {system_info['memory_total_gb']} GB")
            st.write(f"**사용 가능 메모리:** {system_info['memory_available_gb']} GB")
            
            if system_info["disk_usage"]:
                st.markdown("#### 💿 디스크 사용량")
                st.write(f"**총 용량:** {system_info['disk_usage']['total_gb']} GB")
                st.write(f"**사용 중:** {system_info['disk_usage']['used_gb']} GB")
                st.write(f"**여유 공간:** {system_info['disk_usage']['free_gb']} GB")
                st.write(f"**사용률:** {system_info['disk_usage']['percent']}%")
        
        if system_info.get("error"):
            st.error(f"시스템 정보 오류: {system_info['error']}")
    
    with sub_tab3:
        st.subheader("🎓 학습 관리")
        
        training_status = st.session_state.model_manager.get_training_status()
        
        st.info("⚠️ 학습 기능은 향후 구현 예정입니다.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 학습 상태")
            st.write(f"**상태:** {training_status['status']}")
            if training_status["last_trained"]:
                st.write(f"**마지막 학습:** {training_status['last_trained']}")
            if training_status["current_epoch"]:
                st.write(f"**현재 Epoch:** {training_status['current_epoch']}")
            if training_status["best_accuracy"]:
                st.write(f"**최고 정확도:** {training_status['best_accuracy']}%")
        
        with col2:
            st.markdown("### 학습 설정")
            st.selectbox("YOLOv5 모델 크기", ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"], disabled=True)
            st.number_input("Epochs", min_value=1, max_value=1000, value=100, disabled=True)
            st.number_input("Batch Size", min_value=1, max_value=128, value=16, disabled=True)
            
            if st.button("🚫 학습 시작 (비활성화)", disabled=True):
                st.info("학습 기능 준비 중...")
    
    with sub_tab4:
        st.subheader("🔧 유틸리티")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 모델 다운로드")
            model_option = st.selectbox(
                "YOLOv5 모델 선택",
                ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
            )
            
            if st.button("⬇️ 모델 다운로드"):
                with st.spinner(f"{model_option} 다운로드 중..."):
                    result = st.session_state.model_manager.download_yolo_model(model_option)
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
        
        with col2:
            st.markdown("### 🗑️ 캐시 관리")
            
            if st.button("🧹 캐시 정보 확인"):
                result = st.session_state.model_manager.clear_cache()
                if result["success"]:
                    st.info(result["message"])
                    if result["cache_paths"]:
                        st.write("**캐시 경로:**")
                        for path in result["cache_paths"]:
                            st.write(f"• {path}")
                else:
                    st.error(result["message"])
        
        # 상태 리포트 내보내기
        st.markdown("### 📄 상태 리포트")
        if st.button("💾 리포트 생성"):
            yolo_status = st.session_state.model_manager.get_yolo_status(
                st.session_state.fashion_recommender.detector
            )
            clip_status = st.session_state.model_manager.get_clip_status(
                st.session_state.fashion_recommender.analyzer
            )
            system_info = st.session_state.model_manager.get_system_info()
            
            report = st.session_state.model_manager.export_status_report(
                yolo_status, clip_status, system_info
            )
            
            st.download_button(
                label="⬇️ JSON 다운로드",
                data=report,
                file_name=f"fitzy_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            with st.expander("📋 리포트 미리보기"):
                st.code(report, language="json")

if __name__ == "__main__":
    main()
