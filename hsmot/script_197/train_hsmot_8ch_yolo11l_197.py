import comet_ml
import os
from ultralytics import settings
from ultralytics import YOLO
from hsmot.load_multi_channel_pt import load_multi_channel_pt


# 加载yolo11l模型
pt_file = '/data3/litianhao/hsmot/yolo11/yolo11l.pt'
train_cfg = '/data/users/litianhao/ultralytics/hsmot/cfg/8ch.yaml'
data_cfg = '/data/users/litianhao/ultralytics/hsmot/cfg_data/hsmot_8ch.yaml'
model_cfg = '/data/users/litianhao/ultralytics/ultralytics/cfg/models/11/yolo11l-obb-8ch.yaml'

experiment_name = 'yolov11l_8ch_CocoPretrain_CopyFirstConv_imgsize1280_4gpu'


model = YOLO(model_cfg).load(load_multi_channel_pt(pt_file, 8, pt_file.replace('.pt', '_8ch.pt'), version='RGBRGB'))


# # Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(
    data=data_cfg,
    epochs=50, 
    device = [4,5,6,7],
    project = '/data3/litianhao/hsmot/yolo11',
    task = "obb",
    name = experiment_name,
    batch = 8,
    imgsz = 1280,
    cfg = train_cfg,
    workers = 0,
    )