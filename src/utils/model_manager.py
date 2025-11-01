"""
모델 상태 관리자
YOLOv5와 CLIP 모델의 상태를 확인하고 관리하는 유틸리티
"""

import os
import torch
import psutil
from pathlib import Path
from datetime import datetime
import json

class ModelManager:
    """모델 상태 및 관리 클래스"""
    
    def __init__(self):
        self.models_dir = Path("models/weights")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def get_yolo_status(self, detector=None):
        """YOLOv5 모델 상태 확인"""
        status = {
            "loaded": False,
            "model_path": None,
            "model_name": None,
            "model_size": None,
            "config_path": None,
            "available_models": [],
            "error": None
        }
        
        try:
            if detector and hasattr(detector, 'model'):
                status["loaded"] = True
                # YOLO 모델 정보 추출
                model_path = None
                if hasattr(detector.model, 'ckpt_path'):
                    model_path = detector.model.ckpt_path
                elif hasattr(detector.model, 'path'):
                    model_path = detector.model.path
                
                status["model_path"] = str(model_path) if model_path else "yolov5n.pt (기본)"
                status["model_name"] = os.path.basename(str(model_path)) if model_path else "yolov5n.pt"
                
                # 모델 파일 크기 확인
                if model_path and os.path.exists(model_path):
                    status["model_size"] = self._format_size(os.path.getsize(model_path))
            
            # 사용 가능한 모델 목록 확인
            available = ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
            status["available_models"] = available
            
            # 커스텀 모델 확인
            if self.models_dir.exists():
                custom_models = list(self.models_dir.glob("*.pt"))
                status["available_models"].extend([f.name for f in custom_models])
            
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def get_clip_status(self, analyzer=None):
        """CLIP 모델 상태 확인"""
        status = {
            "loaded": False,
            "model_name": None,
            "model_size": None,
            "device": None,
            "device_type": None,
            "memory_usage": None,
            "config": None,
            "error": None
        }
        
        try:
            if analyzer and hasattr(analyzer, 'model'):
                status["loaded"] = True
                status["model_name"] = getattr(analyzer, 'model_name', 'openai/clip-vit-base-patch32')
                status["device"] = analyzer.device
                status["device_type"] = "GPU" if torch.cuda.is_available() and analyzer.device == "cuda" else "CPU"
                
                # 모델 파라미터 수 계산
                if analyzer.model:
                    total_params = sum(p.numel() for p in analyzer.model.parameters())
                    status["config"] = {
                        "total_parameters": f"{total_params:,}",
                        "trainable_parameters": sum(p.numel() for p in analyzer.model.parameters() if p.requires_grad)
                    }
                
                # GPU 메모리 사용량
                if torch.cuda.is_available() and analyzer.device == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    status["memory_usage"] = {
                        "allocated_gb": f"{memory_allocated:.2f}",
                        "reserved_gb": f"{memory_reserved:.2f}"
                    }
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def get_system_info(self):
        """시스템 정보 가져오기"""
        info = {
            "python_version": None,
            "pytorch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "cpu_count": None,
            "memory_total_gb": None,
            "memory_available_gb": None,
            "disk_usage": None
        }
        
        try:
            import sys
            import platform
            info["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
            
            # CPU 및 메모리 정보
            info["cpu_count"] = psutil.cpu_count()
            memory = psutil.virtual_memory()
            info["memory_total_gb"] = f"{memory.total / 1024**3:.2f}"
            info["memory_available_gb"] = f"{memory.available / 1024**3:.2f}"
            
            # 디스크 사용량
            disk = psutil.disk_usage('/')
            info["disk_usage"] = {
                "total_gb": f"{disk.total / 1024**3:.2f}",
                "used_gb": f"{disk.used / 1024**3:.2f}",
                "free_gb": f"{disk.free / 1024**3:.2f}",
                "percent": f"{disk.percent:.1f}"
            }
        except Exception as e:
            info["error"] = str(e)
        
        return info
    
    def get_training_status(self):
        """학습 상태 확인 (향후 구현)"""
        return {
            "status": "not_started",
            "last_trained": None,
            "training_history": [],
            "current_epoch": None,
            "best_accuracy": None
        }
    
    def download_yolo_model(self, model_name="yolov5n.pt"):
        """YOLOv5 모델 다운로드"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_name)
            return {
                "success": True,
                "message": f"모델 {model_name} 다운로드 완료",
                "path": str(model.ckpt_path) if hasattr(model, 'ckpt_path') else model_name
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"다운로드 실패: {str(e)}"
            }
    
    def clear_cache(self):
        """모델 캐시 정리"""
        try:
            cache_paths = [
                Path.home() / ".cache" / "huggingface",
                Path.home() / ".cache" / "ultralytics"
            ]
            
            cleared = []
            for cache_path in cache_paths:
                if cache_path.exists():
                    cleared.append(str(cache_path))
            
            return {
                "success": True,
                "message": "캐시 경로 확인 완료",
                "cache_paths": cleared
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"캐시 정리 실패: {str(e)}"
            }
    
    def _format_size(self, size_bytes):
        """바이트를 읽기 쉬운 형태로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def export_status_report(self, yolo_status, clip_status, system_info):
        """상태 리포트를 JSON으로 내보내기"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "yolo": yolo_status,
            "clip": clip_status,
            "system": system_info
        }
        return json.dumps(report, indent=2, ensure_ascii=False)

