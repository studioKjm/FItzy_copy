"""
Fitzy 패션 코디 추천 앱 - 메인 애플리케이션
Streamlit 기반 웹 인터페이스
"""

import streamlit as st
import datetime
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
        
        # 자동 인식된 성별이 있으면 업데이트 (하지만 수동 변경도 허용)
        if 'auto_gender' in st.session_state and st.session_state.auto_gender:
            gender_index_map = {"남성": 0, "여성": 1, "공용": 2}
            auto_index = gender_index_map.get(st.session_state.auto_gender, st.session_state.selected_gender)
            # 자동 인식 성별과 현재 선택이 다르면 자동 인식값으로 업데이트
            if st.session_state.selected_gender != auto_index:
                st.session_state.selected_gender = auto_index
        
        gender = st.selectbox("성별", gender_options, index=st.session_state.selected_gender, key="gender_selectbox")
        
        # 수동 선택 시 업데이트
        if gender != gender_options[st.session_state.selected_gender]:
            st.session_state.selected_gender = gender_options.index(gender)
        
        # 자동 인식 성별 표시
        if 'auto_gender' in st.session_state and st.session_state.auto_gender:
            if gender == st.session_state.auto_gender:
                st.info(f"✅ 자동 인식: {st.session_state.auto_gender}")
            else:
                st.warning(f"🤖 자동 인식: {st.session_state.auto_gender} (현재: {gender})")

        # 진단 모드
        debug_mode = st.toggle("🔍 진단 모드 (YOLO/CLIP 상세 분석)", value=False)

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
                    st.session_state.auto_gender = detected_gender
                    gender_index_map = {"남성": 0, "여성": 1, "공용": 2}
                    st.session_state.selected_gender = gender_index_map.get(detected_gender, 0)
                st.session_state.last_gender_hash = current_image_hash
            
            # 외모 및 패션 점수 계산
            appearance_scores = st.session_state.scoring_system.score_appearance(face_info, body_info)
            fashion_scores = st.session_state.scoring_system.score_fashion(
                result.get("detected_items", {}).get("items", []),
                result.get("style_analysis", {}),
                weather,
                season,
                temperature  # 온도 파라미터 추가
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
    
    recommendations = st.session_state.recommendation_engine.get_personalized_recommendation(mbti, temp, weather, season)

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
                        st.write(f"{i}. **{d.get('class','?')}** (신뢰도: {d.get('confidence',0):.2f})")
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
    
    # 3가지 버전 코디 추천
    col1, col2, col3 = st.columns(3)
    
    outfit_styles = ["캐주얼", "포멀", "트렌디"]
    outfit_descriptions = [
        f"{recommendations['mbti_style']['style']} 스타일",
        f"{recommendations['seasonal_info']['mood']}한 {recommendations['seasonal_info']['materials'][0]} 소재",
        f"{recommendations['weather_info']['mood']}한 스타일"
    ]
    
    for idx, (col, style, desc) in enumerate(zip([col1, col2, col3], outfit_styles, outfit_descriptions)):
        with col:
            st.write(f"**추천 코디 {idx+1}**")
            st.write(f"**{style} 스타일**")
            st.info(desc)
            st.write(f"**아이템:**")
            if idx == 0:
                st.write(f"• {recommendations['mbti_style']['colors'][0]} 상의")
                st.write(f"• {recommendations['seasonal_info']['colors'][0]} 하의")
            elif idx == 1:
                st.write(f"• {recommendations['seasonal_info']['materials'][0]} 재킷")
                st.write(f"• {recommendations['seasonal_info']['colors'][0]} 바지")
            else:
                st.write(f"• {recommendations['weather_info']['accessories'][0]}")
                st.write(f"• {recommendations['temperature_guidance']['material']} 재킷")
            # 구체 제품 추천
            products = st.session_state.recommendation_engine.recommend_products(style, gender)
            st.write("**추천 제품:**")
            for p in products:
                st.write(f"• {p}")
    
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
