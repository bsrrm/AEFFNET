import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'D:/Users/Desktop/help/NEW/ptune-prune21/weights/prune.pt',
        'data':'D:/Users/Desktop/yolov11/ultralytics-yolo11-main/dataset/dataset final/data.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 32,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project':'runs/distill',
        'name':'yoloaff',
        
        # distill
        'prune_model': True,
        'teacher_weights': 'D:/Users/Desktop/yolov11/ultralytics-yolo11-main/self/BBB/400for2/weights/best.pt',
        'teacher_cfg': 'D:/Users/Desktop/yolov11/ultralytics-yolo11-main/ultralytics/cfg/models/11/',
        'kd_loss_type': 'logical',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '18,21,24',
        'student_kd_layers': '18,21,24',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()