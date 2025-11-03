# 가상환경 생성
python -m venv fitzy_env

# 가상환경 활성화
fitzy_env\Scripts\activate

# mac
source fitzy_env/bin/activate

# 가상환경 비활성화
deactivate

# requirements.txt로 한번에 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run app.py

# 패키지 목록 확인
pip list

# requirements.txt 업데이트
pip freeze > requirements.txt

# 가상환경이 활성화되었는지 확인
which python

# 패키지 설치 확인
pip show streamlit

