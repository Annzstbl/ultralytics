# import comet_ml
import os
from ultralytics import settings
from ultralytics import YOLO
# from hsmot.load_multi_channel_pt import load_multi_channel_pt


# 加载yolo11l模型
pt_file = '/data3/litianhao/hsmot/yolo11/yolo11l.pt'
train_cfg = '/data/users/wangying01/lth/hsmot/ultralytics/hsmot/cfg/rgb.yaml'
data_cfg = '/data/users/wangying01/lth/hsmot/ultralytics/hsmot/cfg_data_rgb_99/hsmot_rgb.yaml'
model_cfg = '/data/users/wangying01/lth/hsmot/ultralytics/ultralytics/cfg/models/11/yolo11l-obb.yaml'

experiment_name = '99debug/yolov11l_rgb_CocoPretrain_imgsize1280_1gpu' 


# model = YOLO(model_cfg).load(load_multi_channel_pt(pt_file, 8, pt_file.replace('.pt', '_8ch.pt'), version='RGBRGB'))
model = YOLO(model_cfg).load(pt_file)


results = model.train(
    data=data_cfg,
    epochs=50, 
    device = [2],
    project = '/data3/litianhao/hsmot/yolo11',
    task = "obb",
    name = experiment_name,
    batch = 4,
    imgsz = 1280,
    cfg = train_cfg,
    workers = 2,
    )