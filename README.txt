1. 프로젝트 주제
패션 코디 추천 앱 Fitzy
본 프로젝트는 사용자가 업로드한 옷 이미지를 바탕으로, AI모델을 활용하여 자동으로 옷
패션을 탐지하고, 색상과 스타일을 분석하여 최적의 코디를 추천해주는 패션 추천시스템
제작이 프로젝트 주제입니다. 이 시스템은 실생활 패션 관리, 온라인 쇼핑몰 추천, 스타
일 및 코디 컨설팅 등 다양한 분야에서 활용 가능하며, 특히 실생활에서 쉽게 활용 가능
한 실용적 앱입니다.

2. 프로젝트 목적 및 최종예상결과물
프로젝트 목적
- 현대인의 패션 선택 과정은 다양성과 복잡성으로 인해 많은 시간과 노력이 필요
합니다. 본 프로젝트는 AI 기반 이미지 분석과 텍스트 이미지 매칭 기술을 활용
하여, 사용자가 업로드한 옷 이미지를 자동으로 분석하고, 최적의 코디를 추천함
으로써 패션 선택 과정을 효율적으로 지원합니다.
- 프로젝트에서는 YOLOv5와 CLIP 두 개의 서로 다른 오픈소스 모델을 활용함으로
써, 학습, 탐지, 추천 기능을 모두 포함한 완전한 실용 앱을 개발할 예정입니다.

최종 예상 결과물
1. 웹/데스크탑 UI: 사용자가 옷 이미지를 업로드하고 추천 코디를 확인할 수 있는
직관적인 인터페이스 제공
2. 이미지 분석 기능: YOLOv5 모델을 사용하여 상의, 하의, 신발 등 개별 옷 아이템
탐지
3. 스타일 분석 및 추천 기능: CLIP 모델을 사용하여 탐지된 아이템의 색상, 패턴, 스
타일을 분석하고, 데이터셋 기반으로 최적 코디 조합 추천
4. 추천 결과 시각화: 추천된 코디 이미지와 설명, 색상과 스타일 키워드 제공
5. 성능 보고서: 탐지 정확도, 추천 품질, 모델 처리 속도 등의 지표를 포함한 테스
트 결과 보고서 제공
6. 확장 가능성: 이후 기능 확장 시, 사용자의 취향기반 개인화 추천, 시즌별 추천,
스타일 트렌드 반영 등 추가 기능 구현 가능

3. 프로젝트에 활용할 오픈소스 개요 (오픈소스 소개, 활용 목적)
YOLOv5
- 객체 탐지 분야에서 널리 사용되는 딥러닝 모델로, 이미지 내 객체 위치와 클래
스를 빠르고 정확하게 탐지 가능. 학습·추론 속도가 빠르고, 다양한 커스텀 데이
터셋에 적용 가능
- 사용자가 업로드한 옷 이미지에서 상의, 하의, 신발 등 개별 아이템을 탐지

CLIP
- OpenAI에서 개발한 모델로, 이미지와 텍스트를 동일한 임베딩 공간으로 매핑하
여 의미 기반 검색, 분류, 이미지-텍스트 매칭 가능
- YOLOv5로 탐지된 옷 아이템의 스타일, 색상, 패턴 분석 → 데이터셋 기반 코디
추천

활용 방법 상세
1. YOLOv5로 업로드된 옷 이미지에서 각 아이템 영역 탐지
2. 탐지된 아이템 이미지를 CLIP 모델에 입력하여 스타일 및 색상 분석
3. 데이터셋 내 유사 스타일의 다른 아이템과 매칭 → 최적 코디 후보 생성
4. 추천 코디를 시각화하여 UI에 출력

개발 언어 및 프레임워크 후보
- Python
- PyTorch
- Streamlit/Flask
- OpenCV
모델 학습 및 추론 환경
- 로컬 PC GPU
데이터셋
- Fashion-MNIST
- COCO

- 커스텀 패션 이미지 데이터셋 등
기능별 개발 세부 사항
- YOLOv5 모델 학습 및 fine-tuning: 사용자 커스텀 옷 이미지 학습
- CLIP 모델 활용: 탐지된 아이템의 이미지-텍스트 매칭
- UI: 이미지 업로드, 추천 코디 시각화, 다운로드 기능
테스트 환경
- 로컬 이미지 업로드 → 탐지 정확도 및 추천 품질 검증
협업 및 버전 관리
- Git/GitHub 사용, 코드 병합 및 팀 협업

4. 참고문헌
1. Bochkovskiy, Alexey, et al. “YOLOv4: Optimal Speed and Accuracy of Object Detection.”
arXiv preprint arXiv:2004.10934 (2020).
2. Jocher, Glenn, et al. “Ultralytics YOLOv5.” GitHub repository,
https://github.com/ultralytics/yolov5</path></svg>
3. Radford, Alec, et al. “Learning Transferable Visual Models From Natural Language
Supervision.” arXiv preprint arXiv:2103.00020 (2021).
4. OpenAI. “CLIP: Connecting Text and Images.” GitHub repository,
https://github.com/openai/CLIP</path></svg>

5. Fashion-MNIST Dataset, https://github.com/zalandoresearch/fashion-
mnist</path></svg>

6. OpenCV, https://opencv.org/</path></svg>
7. PyTorch Documentation, https://pytorch.org/docs/stable/index.html</path></svg>
8. Streamlit Documentation, https://docs.streamlit.io/</path></svg>