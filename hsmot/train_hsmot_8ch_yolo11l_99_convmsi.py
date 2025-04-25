# import comet_ml
import os
from ultralytics import settings
from ultralytics import YOLO
from hsmot.load_multi_channel_pt import load_multi_channel_pt, load_convhsi_pt


# 加载yolo11l模型
pt_file = '/data3/litianhao/hsmot/yolo11/yolo11l.pt'
train_cfg = '/data/users/wangying01/lth/hsmot/ultralytics/hsmot/cfg/8ch.yaml'
data_cfg = '/data/users/wangying01/lth/hsmot/ultralytics/hsmot/cfg_data_99/hsmot_8ch.yaml'
model_cfg = '/data/users/wangying01/lth/hsmot/ultralytics/ultralytics/cfg/models/11/yolo11l-obb-8ch-convmsi.yaml'

experiment_name = '99/yolov11l_8ch_CocoPretrain_convhsi_imgsize1280_1gpu' 


model = YOLO(model_cfg).load(load_convhsi_pt(pt_file, pt_file.replace('.pt', 'convhsi.pt')))
# model = YOLO(model_cfg).load(pt_file)


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