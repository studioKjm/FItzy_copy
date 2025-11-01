"""
YOLOv5 패션 모델 학습 스크립트
DeepFashion2 데이터셋을 사용하여 패션 전용 YOLOv5 모델 학습
"""

import os
import sys
from ultralytics import YOLO
from pathlib import Path
import torch

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
    checkpoint_abs_path = None
    if resume and resume_from:
        # 이어서 학습: 체크포인트에서 로드
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists():
            # 상대 경로로 시도
            checkpoint_path = BASE_DIR / resume_from
        if not checkpoint_path.exists():
            print(f"❌ 오류: 체크포인트를 찾을 수 없습니다: {resume_from}")
            return False
        
        # 절대 경로로 변환
        checkpoint_abs_path = checkpoint_path.resolve()
        
        print(f"\n🔄 이어서 학습: {checkpoint_abs_path}")
        try:
            # 체크포인트 파일 내부의 경로 정보 수정
            # 체크포인트는 torch로 로드하여 메타데이터 수정
            # weights_only=False: PyTorch 2.6+ 기본값 변경으로 인해 명시적으로 False 설정 필요
            ckpt = torch.load(checkpoint_abs_path, map_location='cpu', weights_only=False)
            
            # 체크포인트 내부의 모든 경로 정보를 현재 경로로 수정
            # 체크포인트의 모든 키 확인 및 경로 수정
            current_project = str(BASE_DIR / 'runs' / 'train')
            current_name = 'yolo5_fashion2'
            current_save_dir = str(BASE_DIR / 'runs' / 'train' / 'yolo5_fashion2')
            
            # 체크포인트의 최상위 레벨 키 수정
            for key in ['save_dir', 'project', 'name', 'wdir']:
                if key in ckpt:
                    if key == 'save_dir' or key == 'wdir':
                        ckpt[key] = current_save_dir
                    elif key == 'project':
                        ckpt[key] = current_project
                    elif key == 'name':
                        ckpt[key] = current_name
            
            # train_args 딕셔너리 수정 (가장 중요!)
            if 'train_args' in ckpt and isinstance(ckpt['train_args'], dict):
                train_args_dict = ckpt['train_args']
                train_args_dict['project'] = current_project
                train_args_dict['name'] = current_name
                if 'save_dir' in train_args_dict:
                    train_args_dict['save_dir'] = current_save_dir
                if 'wdir' in train_args_dict:
                    train_args_dict['wdir'] = current_save_dir
                print(f"✅ train_args의 경로 정보를 수정했습니다.")
            
            # args 정보도 수정 (dict 또는 객체일 수 있음)
            if 'args' in ckpt and ckpt['args'] is not None:
                args = ckpt['args']
                if isinstance(args, dict):
                    for key in ['project', 'name', 'save_dir', 'wdir']:
                        if key in args:
                            if key == 'save_dir' or key == 'wdir':
                                args[key] = current_save_dir
                            elif key == 'project':
                                args[key] = current_project
                            elif key == 'name':
                                args[key] = current_name
                else:
                    # 객체인 경우
                    for attr in ['project', 'name', 'save_dir', 'wdir']:
                        if hasattr(args, attr):
                            if attr == 'save_dir' or attr == 'wdir':
                                setattr(args, attr, current_save_dir)
                            elif attr == 'project':
                                setattr(args, attr, current_project)
                            elif attr == 'name':
                                setattr(args, attr, current_name)
            
            # 체크포인트 내부의 모든 문자열 값에서 맥북 경로를 찾아서 교체
            # 텐서나 복잡한 객체는 건드리지 않고 문자열만 수정
            import torch as torch_module
            
            def safe_replace_paths(obj, max_depth=10, current_depth=0):
                """안전하게 딕셔너리/리스트 내의 문자열 경로만 교체 (텐서는 건드리지 않음)"""
                if current_depth > max_depth:
                    return obj
                
                # 텐서나 numpy 배열은 건드리지 않음
                if isinstance(obj, (torch_module.Tensor, torch_module.nn.Module)):
                    return obj
                try:
                    import numpy as np
                    if isinstance(obj, np.ndarray):
                        return obj
                except:
                    pass
                
                if isinstance(obj, str):
                    # 문자열인 경우에만 경로 교체
                    if '/Users/jimin' in obj or 'C:\\Users\\jimin' in obj:
                        obj = obj.replace('/Users/jimin/opensw/FItzy_copy', str(BASE_DIR))
                        obj = obj.replace('C:\\Users\\jimin\\opensw\\FItzy_copy', str(BASE_DIR))
                        obj = obj.replace('/Users/jimin', str(BASE_DIR.parent.parent / 'jimin'))
                        obj = obj.replace('C:\\Users\\jimin', str(BASE_DIR.parent.parent / 'jimin'))
                    return obj
                elif isinstance(obj, dict):
                    # 딕셔너리: 값만 재귀적으로 처리
                    return {k: safe_replace_paths(v, max_depth, current_depth + 1) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    # 리스트/튜플: 각 항목을 재귀적으로 처리
                    result = [safe_replace_paths(item, max_depth, current_depth + 1) for item in obj]
                    return type(obj)(result) if isinstance(obj, tuple) else result
                else:
                    # 다른 타입은 그대로 반환 (텐서, 모델 등)
                    return obj
            
            # 체크포인트 전체에서 경로 교체 (안전한 방법)
            ckpt = safe_replace_paths(ckpt)
            
            # scaler 상태 확인 및 수정 (비어있거나 손상된 경우 복구)
            if 'scaler' in ckpt:
                scaler_state = ckpt['scaler']
                # scaler가 비어있거나 None인 경우 새로 초기화
                if scaler_state is None or (isinstance(scaler_state, dict) and len(scaler_state) == 0):
                    # AMP를 사용하는 경우 scaler 초기화
                    from torch.cuda.amp import GradScaler
                    new_scaler = GradScaler()
                    ckpt['scaler'] = new_scaler.state_dict()
                    print(f"⚠️ 체크포인트의 scaler가 비어있어 새로 초기화했습니다.")
                elif isinstance(scaler_state, dict):
                    # scaler가 딕셔너리인 경우, 필수 키 확인
                    required_keys = ['scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker']
                    if not all(key in scaler_state for key in required_keys):
                        from torch.cuda.amp import GradScaler
                        new_scaler = GradScaler()
                        ckpt['scaler'] = new_scaler.state_dict()
                        print(f"⚠️ 체크포인트의 scaler가 손상되어 새로 초기화했습니다.")
            
            # 수정된 체크포인트를 임시 파일로 저장
            temp_ckpt_path = BASE_DIR / 'temp_checkpoint.pt'
            torch.save(ckpt, temp_ckpt_path)
            
            # 수정된 체크포인트에서 모델 로드
            model = YOLO(str(temp_ckpt_path))
            
            # 모델이 로드된 후에도 trainer 초기화 시 경로를 강제로 설정
            # model.train() 호출 전에 체크포인트 내부 경로 정보를 완전히 무시하도록 설정
            print(f"✅ 체크포인트 로드 완료: {checkpoint_abs_path}")
            print(f"📝 경로 정보를 현재 환경에 맞게 수정했습니다.")
            
            # 원본 체크포인트 경로 업데이트 (나중에 삭제할 임시 파일)
            checkpoint_abs_path = temp_ckpt_path
            
            # 체크포인트 메타데이터에 명시적으로 현재 경로 저장
            # Ultralytics가 resume할 때 체크포인트의 경로 대신 우리가 지정한 경로를 사용하도록 함
            global_checkpoint_path = checkpoint_abs_path
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            import traceback
            traceback.print_exc()
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
                'name': 'yolo5_fashion2',  # 실제 경로와 일치시키기
                'project': str(BASE_DIR / 'runs' / 'train'),  # 절대 경로로 지정 (체크포인트의 경로 덮어쓰기)
                'patience': 50,  # Early stopping
                'save': True,
                'val': True,
                'device': device,
                'workers': 4 if device != "cpu" else 0,  # CPU는 멀티프로세싱 비권장
                'exist_ok': True,  # 기존 디렉토리 허용
            }
            
            # 이어서 학습인 경우: 체크포인트 절대 경로를 resume로 전달
            if resume and checkpoint_abs_path:
                # 디렉토리 변경 후에도 절대 경로는 유효함
                # project와 name을 명시적으로 지정하여 체크포인트의 이전 경로를 덮어쓰기
                # 중요: resume 경로는 절대 경로로 전달하되, project와 name으로 출력 경로를 완전히 덮어쓰기
                train_args['resume'] = str(checkpoint_abs_path)
                # 체크포인트 내부의 경로를 완전히 무시하고 강제로 현재 경로 사용
                train_args['project'] = str(BASE_DIR / 'runs' / 'train')
                train_args['name'] = 'yolo5_fashion2'
                # 추가로 override 옵션을 사용하여 모든 경로 관련 설정 덮어쓰기
                print(f"📌 이어서 학습 모드: 체크포인트에서 재개")
                print(f"📁 출력 경로: {train_args['project']}/{train_args['name']}")
                print(f"⚠️ 체크포인트 내부 경로를 무시하고 위 경로를 사용합니다.")
            
            results = model.train(**train_args)
        finally:
            os.chdir(original_dir)  # 원래 디렉토리로 복귀
        
        print(f"\n✅ 학습 완료!")
        print(f"📁 결과 저장 위치: {results.save_dir}")
        
        # 임시 체크포인트 파일 삭제 (있는 경우)
        temp_ckpt = BASE_DIR / 'temp_checkpoint.pt'
        if temp_ckpt.exists():
            try:
                temp_ckpt.unlink()
                print(f"🗑️ 임시 체크포인트 파일 삭제 완료")
            except Exception as e:
                print(f"⚠️ 임시 파일 삭제 실패 (무시 가능): {e}")
        
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

