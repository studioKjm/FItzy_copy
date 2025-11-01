"""
YOLOv5 패션 모델 학습 스크립트
DeepFashion2 데이터셋을 사용하여 패션 전용 YOLOv5 모델 학습
"""

import os
import sys
from ultralytics import YOLO
from pathlib import Path

# 프로젝트 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "deepfashion2_data"
DATA_YAML = DATA_DIR / "data.yaml"
MODELS_DIR = BASE_DIR / "models" / "weights"

def train_fashion_model(
    model_size="n",  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    device="cpu",  # "cpu" or 0 (GPU)
    resume=False,  # 이어서 학습 여부
    resume_from=None  # 체크포인트 경로 (resume=True일 때)
):
    """
    패션 전용 YOLOv5 모델 학습
    
    Args:
        model_size: 모델 크기 ('n', 's', 'm', 'l', 'x')
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        img_size: 이미지 크기
        device: 학습 장치 ('cpu' or GPU 번호)
        resume: 이어서 학습 여부
        resume_from: 체크포인트 경로 (예: 'runs/train/yolov5_fashion2/weights/last.pt')
    """
    
    # 데이터셋 확인
    if not DATA_YAML.exists():
        print(f"❌ 오류: 데이터셋 설정 파일을 찾을 수 없습니다: {DATA_YAML}")
        print(f"   데이터셋 경로: {DATA_DIR}")
        return False
    
    if not (DATA_DIR / "train").exists():
        print(f"❌ 오류: 학습 데이터를 찾을 수 없습니다: {DATA_DIR / 'train'}")
        return False
    
    print(f"✅ 데이터셋 확인 완료: {DATA_DIR}")
    print(f"📄 설정 파일: {DATA_YAML}")
    
    # 모델 로드
    if resume and resume_from:
        # 이어서 학습: 체크포인트에서 로드
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists():
            # 상대 경로로 시도
            checkpoint_path = BASE_DIR / resume_from
        if not checkpoint_path.exists():
            print(f"❌ 오류: 체크포인트를 찾을 수 없습니다: {resume_from}")
            return False
        
        print(f"\n🔄 이어서 학습: {checkpoint_path}")
        try:
            model = YOLO(str(checkpoint_path))
            print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            return False
    else:
        # 처음부터 학습: 사전 학습 모델 로드
        model_name = f"yolov5{model_size}.pt"
        print(f"\n🚀 모델 로드 중: {model_name}")
        
        try:
            model = YOLO(model_name)
            print(f"✅ 모델 로드 완료: {model_name}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    # 학습 시작
    print(f"\n📊 학습 설정:")
    if resume and resume_from:
        print(f"   - 모드: 이어서 학습")
        print(f"   - 체크포인트: {resume_from}")
    else:
        print(f"   - 모델: {model_name}")
    print(f"   - 데이터셋: {DATA_YAML}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Image Size: {img_size}")
    print(f"   - Device: {device}")
    print(f"\n🎯 학습 시작...\n")
    
    try:
        # YOLO는 data.yaml의 path 필드를 기준으로 경로를 해석합니다
        # data.yaml이 있는 디렉토리를 작업 디렉토리로 변경
        original_dir = os.getcwd()
        os.chdir(DATA_DIR)  # deepfashion2_data 디렉토리로 이동
        
        try:
            train_args = {
                'data': str(DATA_YAML.name),  # 파일명만 전달 (현재 디렉토리 기준)
                'epochs': epochs,
                'imgsz': img_size,
                'batch': batch_size,
                'name': 'yolov5_fashion',
                'project': str(BASE_DIR / 'runs' / 'train'),  # 절대 경로로 지정
                'patience': 50,  # Early stopping
                'save': True,
                'val': True,
                'device': device,
                'workers': 4 if device != "cpu" else 0,  # CPU는 멀티프로세싱 비권장
            }
            
            # 이어서 학습인 경우 resume 옵션 추가
            if resume:
                train_args['resume'] = True
                print(f"📌 이어서 학습 모드: 체크포인트에서 재개")
            
            results = model.train(**train_args)
        finally:
            os.chdir(original_dir)  # 원래 디렉토리로 복귀
        
        print(f"\n✅ 학습 완료!")
        print(f"📁 결과 저장 위치: {results.save_dir}")
        
        # 학습된 모델을 프로젝트 모델 디렉토리로 복사
        # results.save_dir은 절대 경로로 반환됨
        best_model = Path(results.save_dir) / "weights" / "best.pt"
        target_model = BASE_DIR / MODELS_DIR / "yolov5_fashion.pt"  # 절대 경로 사용
        
        if best_model.exists():
            target_model.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(best_model, target_model)
            print(f"✅ 모델 복사 완료: {target_model}")
            print(f"\n🎉 앱에서 자동으로 패션 전용 모델을 사용합니다!")
        else:
            print(f"⚠️ 최고 모델 파일을 찾을 수 없습니다: {best_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv5 패션 모델 학습")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                       help="모델 크기 (n=nanos, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--img-size", type=int, default=640, help="이미지 크기")
    parser.add_argument("--device", type=str, default="cpu", help="학습 장치 (cpu or 0,1,2...)")
    parser.add_argument("--resume", action="store_true", help="이어서 학습 (체크포인트에서 재개)")
    parser.add_argument("--resume-from", type=str, default=None, 
                       help="체크포인트 경로 (예: runs/train/yolov5_fashion2/weights/last.pt)")
    
    args = parser.parse_args()
    
    # device 파싱
    if args.device.isdigit():
        device = int(args.device)
    else:
        device = args.device
    
    print("=" * 60)
    print("YOLOv5 패션 전용 모델 학습")
    print("=" * 60)
    
    success = train_fashion_model(
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=device,
        resume=args.resume,
        resume_from=args.resume_from
    )
    
    if success:
        print("\n✅ 학습이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 학습 중 오류가 발생했습니다.")
        exit(1)

