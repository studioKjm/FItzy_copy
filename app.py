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
from config import MBTI_STYLES, SEASONAL_GUIDE, WEATHER_GUIDE

# 전역 변수로 추천 엔진 초기화
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = RecommendationEngine()
if 'fashion_recommender' not in st.session_state:
    st.session_state.fashion_recommender = FashionRecommender()
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

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
        uploaded_file = st.file_uploader("옷 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            st.success("이미지 업로드 완료! 분석 중...")
            # 이미지 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_container_width=True)
            # 코디 추천 결과 표시
            display_outfit_recommendations(image, mbti_type, temperature, weather, season)
    
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

def display_outfit_recommendations(image, mbti, temp, weather, season):
    """코디 추천 결과 표시"""
    # 추천 생성
    recommendations = st.session_state.recommendation_engine.get_personalized_recommendation(
        mbti, temp, weather, season
    )
    
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
