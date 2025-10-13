# 가상환경 생성
python -m venv fitzy_env

# 가상환경 활성화
fitzy_env\Scripts\activate

# 가상환경 비활성화
deactivate

# requirements.txt로 한번에 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run app.py

# 특정 포트로 실행
streamlit run app.py --server.port 8502

# 브라우저 자동 열기 비활성화
streamlit run app.py --server.headless true

# 코드 포맷팅 (Black)
black .

# 테스트 실행
pytest

# 패키지 목록 확인
pip list

# requirements.txt 업데이트
pip freeze > requirements.txt

# 가상환경이 활성화되었는지 확인
which python

# 패키지 설치 확인
pip show streamlit

# 포트 충돌 시 다른 포트 사용
streamlit run app.py --server.port 8502