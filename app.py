"""
Fitzy 패션 코디 추천 앱 - 메인 애플리케이션
Streamlit 기반 웹 인터페이스
"""

import streamlit as st
import datetime
import json
import os
from PIL import Image
from src.utils.recommendation_engine import RecommendationEngine
from src.models.models import FashionRecommender
from src.utils.model_manager import ModelManager
from src.utils.visualization import draw_detections
from src.utils.body_analysis import BodyAnalyzer
from src.utils.scoring_system import ScoringSystem
from src.utils.virtual_fitting import VirtualFittingSystem
from config import MBTI_STYLES

# 설정 파일 경로
SETTINGS_FILE = ".fitzy_settings.json"

def load_settings():
    """설정 파일에서 설정값 로드"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_settings(settings):
    """설정값을 파일에 저장"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"설정 저장 실패: {e}")

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
if 'virtual_fitting' not in st.session_state:
    st.session_state.virtual_fitting = VirtualFittingSystem(
        st.session_state.fashion_recommender.detector,
        st.session_state.fashion_recommender.analyzer
    )

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

# ==================== 공통 UI 함수 ====================

def display_score_metric(label, score, delta_label="점수"):
    """점수 메트릭 표시 (재사용 함수)"""
    st.metric(label, f"{score}/100", 
             delta=f"{score - 70}", 
             delta_color="normal" if score >= 70 else "inverse")
    st.caption(st.session_state.scoring_system.get_score_label(score))

def render_gender_selector():
    """성별 선택 UI 렌더링 (재사용 함수)"""
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
        if st.session_state.selected_gender != auto_index:
            st.session_state.selected_gender = auto_index
    
    # selectbox: 현재 선택된 성별로 표시
    current_selected_index = st.session_state.selected_gender
    gender = st.selectbox(
        "성별", 
        gender_options, 
        index=current_selected_index,
        key=f"gender_selectbox_{current_selected_index}"
    )
    
    # 수동 선택 시 업데이트
    current_selected_gender = gender_options[current_selected_index]
    if gender != current_selected_gender:
        st.session_state.selected_gender = gender_options.index(gender)
    
    # 자동 인식 성별 표시
    if 'auto_gender' in st.session_state and st.session_state.auto_gender:
        if gender == st.session_state.auto_gender:
            st.success(f"✅ 자동 인식: {st.session_state.auto_gender}")
        else:
            st.info(f"🤖 자동 인식: {st.session_state.auto_gender}")
    
    return gender

def render_outfit_items_display(idx, recommendations, image_suggestions, has_image_based, image_based_combinations, temp, gender):
    """코디 아이템 표시 로직 (재사용 함수)"""
    displayed_items = []
    
    if has_image_based and idx < len(image_based_combinations):
        combo = image_based_combinations[idx]
        items = combo.get("items", [])
        for item in items:
            displayed_items.append(item)
            st.write(f"• {item}")
    else:
        # 기존 방식 (템플릿 기반)
        if idx == 0:
            detected_colors = image_suggestions.get("color_matches", {})
            if detected_colors:
                top_color = max(detected_colors.items(), key=lambda x: x[1])[0]
                color_display = top_color
            else:
                color_display = recommendations['mbti_style']['colors'][0]
            
            top_type = "반팔 티셔츠" if temp >= 20 else "긴팔 셔츠"
            item1 = f"{color_display} {top_type}"
            bottom_color = recommendations['seasonal_info']['colors'][0]
            item2 = f"{bottom_color} 바지"
            displayed_items = [item1, item2]
            st.write(f"• {item1}")
            st.write(f"• {item2}")
        elif idx == 1:
            jacket_color = recommendations['seasonal_info']['colors'][0] if recommendations['seasonal_info'].get('colors') else "검은색"
            item1 = f"{jacket_color} {recommendations['seasonal_info']['materials'][0]} 재킷"
            pants_color = "회색" if jacket_color == "검은색" else "베이지"
            item2 = f"{pants_color} 바지"
            displayed_items = [item1, item2]
            st.write(f"• {item1}")
            st.write(f"• {item2}")
        else:
            accessory_color = recommendations['weather_info'].get('colors', ['검은색'])[0] if isinstance(recommendations['weather_info'].get('colors'), list) else "검은색"
            item1 = f"{accessory_color} {recommendations['weather_info']['accessories'][0]}"
            jacket_color = "검은색" if accessory_color == "흰색" else "회색"
            item2 = f"{jacket_color} {recommendations['temperature_guidance']['material']} 재킷"
            displayed_items = [item1, item2]
            st.write(f"• {item1}")
            st.write(f"• {item2}")
    
    return displayed_items

def main():
    """메인 애플리케이션 함수"""
    st.title("👗 Fitzy - AI 패션 코디 추천")
    st.markdown("업로드한 옷 이미지로 최적의 코디를 추천받아보세요!")
    
    # 사이드바 - 사용자 설정
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # MBTI 선택 (모든 16가지 타입)
        mbti_options = [
            "ENFP", "ENTP", "ESFP", "ESTP",
            "ENFJ", "ENTJ", "ESFJ", "ESTJ",
            "INFP", "INFJ", "ISFP", "ISTP",
            "INTP", "INTJ", "ISFJ", "ISTJ"
        ]
        # 설정 파일에서 로드 (서버 재시작 후에도 유지)
        saved_settings = load_settings()
        
        # session_state 초기화 (파일에서 로드한 값으로)
        if 'saved_mbti' not in st.session_state:
            st.session_state.saved_mbti = saved_settings.get('mbti', "ENFP")
        saved_mbti_index = mbti_options.index(st.session_state.saved_mbti) if st.session_state.saved_mbti in mbti_options else 0
        mbti_type = st.selectbox("MBTI 유형", mbti_options, index=saved_mbti_index, key="mbti_selectbox")
        
        # 값이 변경되면 session_state와 파일에 저장
        if st.session_state.saved_mbti != mbti_type:
            st.session_state.saved_mbti = mbti_type
            saved_settings['mbti'] = mbti_type
            save_settings(saved_settings)
        
        # 성별 선택 (자동 인식 기능)
        gender = render_gender_selector()

        # 진단 모드
        if 'saved_debug_mode' not in st.session_state:
            st.session_state.saved_debug_mode = False
        debug_mode = st.toggle("🔍 진단 모드 (YOLO/CLIP 상세 분석)", value=st.session_state.saved_debug_mode, key="debug_mode_toggle")
        st.session_state.saved_debug_mode = debug_mode

        # 날씨 정보 입력
        st.subheader("🌤️ 날씨 정보")
        if 'saved_temperature' not in st.session_state:
            st.session_state.saved_temperature = saved_settings.get('temperature', 20)
        temperature = st.slider("온도 (°C)", -10, 40, st.session_state.saved_temperature, key="temperature_slider")
        if st.session_state.saved_temperature != temperature:
            st.session_state.saved_temperature = temperature
            saved_settings['temperature'] = temperature
            save_settings(saved_settings)
        
        weather_options = ["맑음", "흐림", "비", "눈", "바람"]
        if 'saved_weather' not in st.session_state:
            st.session_state.saved_weather = saved_settings.get('weather', "맑음")
        saved_weather_index = weather_options.index(st.session_state.saved_weather) if st.session_state.saved_weather in weather_options else 0
        weather = st.selectbox("날씨", weather_options, index=saved_weather_index, key="weather_selectbox")
        if st.session_state.saved_weather != weather:
            st.session_state.saved_weather = weather
            saved_settings['weather'] = weather
            save_settings(saved_settings)
        
        # 계절 선택
        season_options = ["봄", "여름", "가을", "겨울"]
        if 'saved_season' not in st.session_state:
            st.session_state.saved_season = saved_settings.get('season', "봄")
        saved_season_index = season_options.index(st.session_state.saved_season) if st.session_state.saved_season in season_options else 0
        season = st.selectbox("계절", season_options, index=saved_season_index, key="season_selectbox")
        if st.session_state.saved_season != season:
            st.session_state.saved_season = season
            saved_settings['season'] = season
            save_settings(saved_settings)
    
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
            
            # 이미지 표시
            st.image(image, caption="업로드된 이미지", width='stretch')
            processed_image = image
            
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
                else:
                    st.warning("⚠️ 체형을 분석할 수 없습니다")
                    st.info(body_info.get("message", "전신 사진을 업로드해주세요."))
            
            # 코디 추천 결과 표시 (원본 이미지 사용, 얼굴/체형 정보 포함)
            # 먼저 YOLO/CLIP 분석 실행 (점수 계산을 위해)
            fr = st.session_state.fashion_recommender
            result = fr.recommend_outfit(processed_image, mbti_type, temperature, weather, season)
            
            # 가상 피팅용 원본 이미지 저장
            user_uploaded_image = image
            # 텍스트 검색에서 사용할 수 있도록 이미지 저장
            st.session_state.user_uploaded_image_for_search = image
            
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
            
            # 점수 표시 (접힌 상태로)
            with st.expander("📊 외모 및 패션 점수", expanded=False):
                col_score1, col_score2 = st.columns(2)
                with col_score1:
                    st.markdown("### 👤 외모 점수")
                    display_score_metric("전체 외모", appearance_scores['전체 외모'])
                
                with col_score2:
                    st.markdown("### 👗 패션 점수")
                    display_score_metric("전체 패션", fashion_scores['전체 패션'])
            
            # 상세 피드백
            feedback = st.session_state.scoring_system.get_detailed_feedback(appearance_scores, fashion_scores, season)
            if feedback:
                with st.expander("💡 개선 제안"):
                    for fb in feedback:
                        st.write(fb)
            
            # 코디 추천 결과 표시 (가상 피팅용 원본 이미지 전달)
            display_outfit_recommendations(
                processed_image, mbti_type, temperature, weather, season, 
                gender, debug_mode, face_info, body_info, original_image=image,
                precomputed_result=result, appearance_scores=appearance_scores, fashion_scores=fashion_scores,
                user_uploaded_image=user_uploaded_image
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
            # 이미지 분석 섹션에서 입력받은 이미지와 세팅값 전달
            user_image = st.session_state.get('user_uploaded_image_for_search', None)
            display_text_search_results(search_query, mbti_type, temperature, weather, season, gender, user_image)
    
    with tab3:
        # 트렌드 및 인기 코디
        st.subheader("🔥 이번 시즌 인기 코디")
        display_trend_outfits(season)
    
    with tab4:
        # 모델 관리 페이지
        display_model_manager()

def display_outfit_recommendations(image, mbti, temp, weather, season, gender, debug_mode=False, 
                                   face_info=None, body_info=None, original_image=None,
                                   precomputed_result=None, appearance_scores=None, fashion_scores=None,
                                   user_uploaded_image=None):
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
    
    # 통합 추천 생성 (성별 + MBTI + 이미지 분석 + 온도/계절 → 스타일 → 아이템 → 제품)
    unified_recommendations = st.session_state.recommendation_engine.generate_unified_outfit_recommendations(
        gender, mbti, temp, weather, season,
        detected_items=detected_items_data.get("items", []),
        style_analysis=style_analysis_data
    )
    
    # 기존 호환성 유지용
    recommendations = st.session_state.recommendation_engine.get_personalized_recommendation(
        mbti, temp, weather, season,
        detected_items=detected_items_data.get("items", []),
        style_analysis=style_analysis_data
    )
    
    # 통합 추천 결과를 기존 recommendations에 병합
    recommendations["outfit_versions"] = unified_recommendations["outfit_versions"]

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
    
    # 통합 추천 결과 사용
    outfit_versions = recommendations.get("outfit_versions", [])
    image_suggestions = recommendations.get("image_suggestions", {})
    
    # 3가지 버전 코디 추천
    col1, col2, col3 = st.columns(3)
    
    # 통합 추천이 있는 경우 사용
    outfit_data_list = []
    outfit_styles = []  # 항상 정의되도록 초기화
    
    if outfit_versions and len(outfit_versions) >= 3:
        # 통합 추천 사용 (성별 + MBTI + 이미지 분석 + 온도/계절)
        for idx, (col, version) in enumerate(zip([col1, col2, col3], outfit_versions[:3])):
            with col:
                st.write(f"**추천 코디 {idx+1}**")
                st.write(f"**{version['style']}**")
                
                st.info(version['description'])
                st.write(f"**아이템:**")
                
                # 아이템 표시
                for item in version['items']:
                    st.write(f"• {item}")
                
                # 추천 제품 표시
                st.write("**추천 제품:**")
                for product in version['products']:
                    st.write(f"• {product}")
                
                # 가상 피팅/AI 생성용 데이터 저장
                outfit_desc = {
                    "items": version['items'],
                    "style": version['style'],
                    "colors": [item.split()[0] for item in version['items'] if item.split()[0] in ["검은색", "흰색", "빨간색", "파란색", "회색", "베이지", "네이비"]][:2],
                    "gender": version['gender']
                }
                current_image_hash = st.session_state.get("last_image_hash", "default")
                cache_key = f"generated_image_{current_image_hash}_{version['style']}_{idx}"
                outfit_data_list.append({
                    "col": col,
                    "outfit_desc": outfit_desc,
                    "style": version['style'],
                    "idx": idx,
                    "cache_key": cache_key
                })
                # outfit_styles에 스타일 추가
                outfit_styles.append(version['style'])
    else:
        # 기존 방식 (하위 호환성)
        style_matches = image_suggestions.get("style_matches", {})
        image_based_combinations = image_suggestions.get("recommendation_based_on_image", [])
        
        if style_matches:
            sorted_styles = sorted(style_matches.items(), key=lambda x: x[1], reverse=True)
            top_styles = [style[0] for style in sorted_styles[:3]]
            outfit_styles_list = []
            for style in ["캐주얼", "포멀", "트렌디"]:
                if style in top_styles:
                    outfit_styles_list.append(style)
            for style in ["캐주얼", "포멀", "트렌디"]:
                if len(outfit_styles_list) < 3 and style not in outfit_styles_list:
                    outfit_styles_list.append(style)
            outfit_styles = outfit_styles_list[:3]
        else:
            outfit_styles = ["캐주얼", "포멀", "트렌디"]
        
        has_image_based = len(image_based_combinations) > 0
        color_suggestions = image_suggestions.get("color_matches", {})
        top_colors = []
        if color_suggestions:
            top_colors = sorted(color_suggestions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        outfit_descriptions = []
        for idx in range(3):
            style = outfit_styles[idx] if idx < len(outfit_styles) else "캐주얼"
            if has_image_based and idx < len(image_based_combinations):
                combo = image_based_combinations[idx]
                reason = combo.get("reason", f"{style} 스타일")
                if top_colors and idx < len(top_colors):
                    reason += f", {top_colors[idx][0]} 톤 추천"
                outfit_descriptions.append(reason)
            else:
                base_desc = ""
                if idx == 0:
                    base_desc = f"{recommendations['mbti_style']['style']} 스타일"
                    if recommendations['mbti_style'].get('colors'):
                        base_desc += f", {recommendations['mbti_style']['colors'][0]} 톤"
                elif idx == 1:
                    base_desc = f"{recommendations['seasonal_info']['mood']}한 {recommendations['seasonal_info']['materials'][0]} 소재"
                    if recommendations['seasonal_info'].get('colors'):
                        base_desc += f", {recommendations['seasonal_info']['colors'][0]} 톤"
                else:
                    base_desc = f"{recommendations['weather_info']['mood']}한 스타일"
                    if top_colors:
                        base_desc += f", {top_colors[0][0]} 톤 추천"
                outfit_descriptions.append(base_desc)
        
        for idx, (col, style, desc) in enumerate(zip([col1, col2, col3], outfit_styles, outfit_descriptions)):
            with col:
                st.write(f"**추천 코디 {idx+1}**")
                st.write(f"**{style} 스타일**")
                st.info(desc)
                st.write(f"**아이템:**")
                
                color_display = None
                if idx == 0:
                    detected_colors = image_suggestions.get("color_matches", {})
                    if detected_colors:
                        color_display = max(detected_colors.items(), key=lambda x: x[1])[0]
                
                displayed_items = render_outfit_items_display(
                    idx, recommendations, image_suggestions, has_image_based, 
                    image_based_combinations, temp, gender
                )
                
                # 아이템 기반으로 제품 추천 (아이템에 맞는 제품 추천)
                mbti_style = recommendations.get('mbti_style', {})
                products = st.session_state.recommendation_engine._generate_product_recommendations(
                    displayed_items, style, gender, mbti_style
                )
                st.write("**추천 제품:**")
                for p in products:
                    st.write(f"• {p}")
                
                outfit_desc = {
                    "items": displayed_items,
                    "style": style,
                    "colors": [color_display] if color_display else recommendations.get('seasonal_info', {}).get('colors', [])[:2],
                    "gender": gender
                }
                current_image_hash = st.session_state.get("last_image_hash", "default")
                cache_key = f"generated_image_{current_image_hash}_{style}_{idx}"
                outfit_data_list.append({
                    "col": col,
                    "outfit_desc": outfit_desc,
                    "style": style,
                    "idx": idx,
                    "cache_key": cache_key
                })
    
    # 모든 코디 텍스트 출력 완료 후 가상 피팅 합성
    if outfit_data_list:
        # 디버깅 정보
        print(f"DEBUG: outfit_data_list 길이: {len(outfit_data_list)}")
        
            # 가상 피팅 모드: 업로드 이미지에 코디 합성
        # 중복 실행 방지: 처리 중인 작업 추적
        processing_key = f"virtual_fitting_processing_{st.session_state.get('last_image_hash', 'default')}"
        
            for data in outfit_data_list:
                with data["col"]:
                # 캐시 키 개선: 아이템 리스트와 성별 포함
                items_str = "_".join(data["outfit_desc"]["items"][:2])  # 상의+하의만
                cache_key = f"virtual_fitting_{data['cache_key']}_{items_str}_{data['outfit_desc']['gender']}"
                    
                    if cache_key not in st.session_state:
                    # 추천 코디 1은 자동 생성, 2와 3은 버튼 클릭으로 생성
                    if data["idx"] == 0:
                        # 추천 코디 1: 자동 생성
                        # 처리 중인지 확인
                        if st.session_state.get(processing_key, False):
                            st.info("⏳ 다른 가상 피팅이 진행 중입니다. 잠시만 기다려주세요...")
                            continue
                        
                        # 처리 시작 플래그 설정
                        st.session_state[processing_key] = True
                        
                        try:
                            # st.spinner 대신 status_placeholder 사용 (다른 탭 블로킹 방지)
                            status_placeholder = st.empty()
                            image_placeholder = st.empty()
                            
                            status_placeholder.info(f"⏳ {data['style']} 스타일 가상 피팅 중...")
                            
                            # 원본 이미지 사용 (user_uploaded_image 또는 image)
                            source_image = user_uploaded_image if user_uploaded_image is not None else image
                            
                            # 가상 피팅 실행
                            fitting_result = st.session_state.virtual_fitting.composite_outfit_on_image(
                                source_image,
                                data["outfit_desc"]["items"],
                                data["outfit_desc"]["gender"]
                            )
                            
                            # fitting_result가 튜플인 경우 (이미지, 프롬프트) 또는 이미지만 반환
                            if isinstance(fitting_result, tuple):
                                fitted_image, prompts_info = fitting_result
                            else:
                                fitted_image = fitting_result
                                prompts_info = []
                            
                            if fitted_image:
                                st.session_state[cache_key] = fitted_image
                                # 프롬프트 정보 캐시
                                prompts_cache_key = f"prompts_{data['cache_key']}_{items_str}_{data['outfit_desc']['gender']}"
                                if prompts_info:
                                    st.session_state[prompts_cache_key] = prompts_info
                                status_placeholder.empty()
                                image_placeholder.image(fitted_image, caption=f"{data['style']} 스타일 가상 피팅", width='stretch')
                                st.success("✅ 가상 피팅 완료")
                                
                                # 프롬프트 표시 (fold 상태)
                                if prompts_info:
                                    with st.expander("📝 사용된 프롬프트 보기", expanded=False):
                                        for idx, prompt_info in enumerate(prompts_info, 1):
                                            st.write(f"**{prompt_info['region']} 영역:**")
                                            st.code(prompt_info['prompt'], language=None)
                            else:
                                status_placeholder.warning("⚠️ 가상 피팅 실패 - 의류 영역을 찾을 수 없습니다")
                        except Exception as e:
                            st.error(f"❌ 가상 피팅 오류: {str(e)}")
                        finally:
                            # 처리 완료 플래그 해제
                            st.session_state[processing_key] = False
                    else:
                        # 추천 코디 2, 3: 버튼 클릭으로 생성
                        button_key = f"generate_fitting_{data['idx']}_{data['cache_key']}"
                        is_processing = st.session_state.get(processing_key, False)
                        
                        if st.button(f"🎨 {data['style']} 스타일 가상 피팅 생성", key=button_key, disabled=is_processing):
                            if is_processing:
                                st.warning("⏳ 다른 가상 피팅이 진행 중입니다. 완료 후 다시 시도해주세요.")
                            else:
                                # 처리 시작 플래그 설정
                                st.session_state[processing_key] = True
                                
                                try:
                                    # st.spinner 대신 status_placeholder 사용 (다른 탭 블로킹 방지)
                                    status_placeholder = st.empty()
                                    image_placeholder = st.empty()
                                    
                                    status_placeholder.info(f"⏳ {data['style']} 스타일 가상 피팅 중...")
                                    
                                    # 원본 이미지 사용 (user_uploaded_image 또는 image)
                                    source_image = user_uploaded_image if user_uploaded_image is not None else image
                                    
                                    # 가상 피팅 실행
                                    fitting_result = st.session_state.virtual_fitting.composite_outfit_on_image(
                                        source_image,
                                        data["outfit_desc"]["items"],
                                        data["outfit_desc"]["gender"]
                                    )
                                    
                                    # fitting_result가 튜플인 경우 (이미지, 프롬프트) 또는 이미지만 반환
                                    if isinstance(fitting_result, tuple):
                                        fitted_image, prompts_info = fitting_result
                                    else:
                                        fitted_image = fitting_result
                                        prompts_info = []
                                    
                                    if fitted_image:
                                        st.session_state[cache_key] = fitted_image
                                        # 프롬프트 정보 캐시
                                        prompts_cache_key = f"prompts_{data['cache_key']}_{items_str}_{data['outfit_desc']['gender']}"
                                        if prompts_info:
                                            st.session_state[prompts_cache_key] = prompts_info
                                        status_placeholder.empty()
                                        image_placeholder.image(fitted_image, caption=f"{data['style']} 스타일 가상 피팅", width='stretch')
                                        st.success("✅ 가상 피팅 완료")
                                        
                                        # 프롬프트 표시 (fold 상태)
                                        if prompts_info:
                                            with st.expander("📝 사용된 프롬프트 보기", expanded=False):
                                                for idx, prompt_info in enumerate(prompts_info, 1):
                                                    st.write(f"**{prompt_info['region']} 영역:**")
                                                    st.code(prompt_info['prompt'], language=None)
                                    else:
                                        status_placeholder.warning("⚠️ 가상 피팅 실패 - 의류 영역을 찾을 수 없습니다")
                                except Exception as e:
                                    st.error(f"❌ 가상 피팅 오류: {str(e)}")
                                finally:
                                    # 처리 완료 플래그 해제
                                    st.session_state[processing_key] = False
                        elif is_processing:
                            st.info("⏳ 다른 가상 피팅이 진행 중입니다. 완료 후 버튼을 클릭하여 생성해주세요.")
                    else:
                        # 캐시된 이미지 사용
                        cached_image = st.session_state[cache_key]
                        st.image(cached_image, caption=f"{data['style']} 스타일 가상 피팅", width='stretch')
                    st.success("✅ 가상 피팅 완료")
                    
                    # 프롬프트 표시 (fold 상태) - 캐시된 프롬프트가 있는 경우
                    prompts_cache_key = f"prompts_{data['cache_key']}_{items_str}_{data['outfit_desc']['gender']}"
                    if prompts_cache_key in st.session_state:
                        with st.expander("📝 사용된 프롬프트 보기", expanded=False):
                            prompts = st.session_state[prompts_cache_key]
                            for idx, prompt_info in enumerate(prompts, 1):
                                st.write(f"**{prompt_info['region']} 영역:**")
                                st.code(prompt_info['prompt'], language=None)
    
    # 추천 이유 및 현재 코디 평가
    st.subheader("💡 이 조합이 어울리는 이유")
    for reason in recommendations['recommendation_reason']:
        st.write(reason)
    
    # 현재 코디 평가 (추천 이유와 연계)
    eval_result = st.session_state.recommendation_engine.evaluate_current_outfit(
        result.get("detected_items", {}).get("items", []),
        result.get("style_analysis", {}),
        weather,
        season
    )
    
    st.markdown("---")
    st.markdown(f"**🧭 현재 코디 평가:** {eval_result['score']} / 100 ({eval_result['label']})")
    for fb in eval_result["feedback"]:
        st.write(f"• {fb}")
    
    # 얼굴/체형 정보 추가 피드백
    if face_info and face_info.get("detected"):
        st.write(f"• 얼굴 형태({face_info.get('face_shape')})에 맞는 넥라인 추천")
    if body_info and body_info.get("detected"):
        st.write(f"• 체형({body_info.get('body_type')})에 최적화된 실루엣 추천")

def display_text_search_results(query, mbti, temperature=None, weather=None, season=None, gender=None, user_image=None):
    """텍스트 검색 결과 표시 및 가상 피팅"""
    from config import SEASONAL_GUIDE  # MBTI_STYLES는 파일 상단에서 이미 import됨
    
    # FashionRecommender의 text_searcher 사용 (성별 전달)
    results = st.session_state.fashion_recommender.text_searcher.search_outfits(query, gender=gender)
    
    st.subheader(f"'{query}' 검색 결과")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**카테고리:** {results.get('category', '일반')}")
        # mood와 colors는 search_outfits에서 반환하지 않으므로 기본값 사용
        mood_map = {
            "파티용": "화려하고 눈에 띄는",
            "출근룩": "전문적이고 세련된",
            "데이트룩": "로맨틱하고 우아한",
            "일반": "편안하고 캐주얼한"
        }
        mood = mood_map.get(results.get('category', '일반'), "편안하고 캐주얼한")
        st.write(f"**무드:** {mood}")
        
        color_map = {
            "파티용": ["빨간색", "검은색", "골드"],
            "출근룩": ["네이비", "회색", "베이지"],
            "데이트룩": ["핑크", "라벤더", "화이트"],
            "일반": ["베이지", "회색", "네이비"]
        }
        colors = color_map.get(results.get('category', '일반'), ["베이지", "회색", "네이비"])
        st.write(f"**추천 색상:** {', '.join(colors)}")
    
    with col2:
        st.write("**추천 아이템:**")
        items = results.get('items', ["캐주얼 웨어"])
        for item in items:
            st.write(f"• {item}")
    
    # MBTI 개인화 적용
    if mbti in MBTI_STYLES:
        st.info(f"💡 {mbti} 유형을 위해 {MBTI_STYLES[mbti]['style']} 요소가 추가로 반영되었습니다.")
    
    # 가상 피팅 이미지 생성 (이미지가 있고 세팅값이 있는 경우)
    if user_image is not None and gender and temperature is not None and weather and season:
        st.markdown("---")
        st.subheader("🎨 가상 피팅 이미지")
        
        # 텍스트 검색 결과를 기반으로 아이템 리스트 생성 (YOLO가 탐지 가능한 형식으로)
        search_items = []
        category = results.get('category', '일반')
        
        # 카테고리별 아이템 생성 (YOLO 탐지 가능한 형식: "색상 타입" 형식)
        # 주의: "셔츠", "바지", "드레스" 등은 YOLO가 탐지 가능한 형식이어야 함
        if category == "파티용":
            if gender == "남성":
                search_items = ["검은색 긴팔 상의", "검은색 바지"]
            else:
                search_items = ["검은색 긴팔 드레스", "검은색 바지"]  # 드레스는 전신이므로 하의는 선택적
        elif category == "출근룩":
            search_items = ["네이비 긴팔 상의", "회색 바지"]
        elif category == "데이트룩":
            if gender == "남성":
                search_items = ["핑크 반팔 상의", "화이트 바지"]
            else:
                search_items = ["핑크 긴팔 드레스", "화이트 바지"]  # 드레스는 전신이므로 하의는 선택적
        else:
            # 일반 카테고리의 경우 MBTI와 계절 기반으로 아이템 생성
            mbti_style = MBTI_STYLES.get(mbti, MBTI_STYLES["ENFP"])
            seasonal_info = SEASONAL_GUIDE.get(season, SEASONAL_GUIDE["봄"])
            
            # 온도 기반 아이템 선택 (YOLO 탐지 가능한 형식)
            if temperature < 15:
                top_color = seasonal_info.get('colors', ['베이지'])[0]
                search_items = [f"{top_color} 긴팔 상의", "회색 바지"]
            else:
                top_color = seasonal_info.get('colors', ['베이지'])[0]
                search_items = [f"{top_color} 반팔 상의", "회색 바지"]
        
        # 이미지 분석 섹션의 가상 피팅 진행 여부 확인
        processing_key = f"virtual_fitting_processing_{st.session_state.get('last_image_hash', 'default')}"
        is_processing = st.session_state.get(processing_key, False)
        
        # 캐시 키
        cache_key = f"text_search_fitting_{query}_{gender}_{hash(str(search_items))}"
        prompts_cache_key = f"text_search_prompts_{query}_{gender}_{hash(str(search_items))}"
        
        # 가상 피팅 버튼
        button_key = f"generate_fitting_{query}_{hash(str(search_items))}"
        
        # 캐시된 이미지가 있으면 표시
        if cache_key in st.session_state:
            cached_image = st.session_state[cache_key]
            st.image(cached_image, caption=f"'{query}' 스타일 가상 피팅", width='stretch')
            st.success("✅ 가상 피팅 완료")
            
            # 프롬프트 표시 (fold 상태)
            if prompts_cache_key in st.session_state:
                with st.expander("📝 사용된 프롬프트 보기", expanded=False):
                    prompts = st.session_state[prompts_cache_key]
                    for idx, prompt_info in enumerate(prompts, 1):
                        st.write(f"**{prompt_info['region']} 영역:**")
                        st.code(prompt_info['prompt'], language=None)
        
        # 버튼 표시 (캐시가 없거나 재생성하고 싶을 때)
        if is_processing:
            st.info("⏳ 이미지 분석 섹션에서 가상 피팅이 진행 중입니다. 완료 후 버튼을 클릭하여 생성해주세요.")
        elif st.button("🎨 가상 피팅 이미지 생성", key=button_key, disabled=is_processing):
            if is_processing:
                st.warning("⏳ 이미지 분석 섹션에서 가상 피팅이 진행 중입니다. 완료 후 다시 시도해주세요.")
            else:
                # 가상 피팅 실행
                with st.spinner(f"🎨 '{query}' 스타일 가상 피팅 중..."):
                    try:
                        fitting_result = st.session_state.virtual_fitting.composite_outfit_on_image(
                            user_image,
                            search_items,
                            gender
                        )
                        
                        # fitting_result가 튜플인 경우 (이미지, 프롬프트) 또는 이미지만 반환
                        if isinstance(fitting_result, tuple):
                            fitted_image, prompts_info = fitting_result
                        else:
                            fitted_image = fitting_result
                            prompts_info = []
                        
                        if fitted_image:
                            st.session_state[cache_key] = fitted_image
                            if prompts_info:
                                st.session_state[prompts_cache_key] = prompts_info
                            st.rerun()  # 페이지 새로고침하여 캐시된 이미지 표시
                        else:
                            st.warning("⚠️ 가상 피팅 실패 - 의류 영역을 찾을 수 없습니다.")
                    except Exception as e:
                        st.error(f"❌ 가상 피팅 오류: {str(e)}")
    elif user_image is None:
        st.info("💡 가상 피팅을 보려면 이미지 분석 섹션에서 먼저 이미지를 업로드해주세요.")
    

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
