# import comet_ml
import os
from ultralytics import settings
from ultralytics import YOLO
# from hsmot.load_multi_channel_pt import load_multi_channel_pt


# 加载yolo11l模型
# pt_file = '/data3/litianhao/hsmot/yolo11/yolo11l.pt'
pt_file = '/data/users/litianhao/hsmot_code/workdir/yolo11/yolov11l_3ch_CocoPretrain_imgsize1280_2gpu/weights/best.pt'
train_cfg = '/data/users/litianhao/hsmot_code/ultralytics/hsmot/cfg/rgb.yaml'
data_cfg = '/data/users/litianhao/hsmot_code/ultralytics/hsmot/cfg_data/hsmot_8ch.yaml'
model_cfg = '/data/users/litianhao/hsmot_code/ultralytics/ultralytics/cfg/models/11/yolo11l-obb.yaml'

# experiment_name = 'yolov11l_3ch_CocoPretrain_imgsize1280_2gpu_val' 
experiment_name = 'debug/val' 


# model = YOLO(model_cfg).load(load_multi_channel_pt(pt_file, 8, pt_file.replace('.pt', '_8ch.pt'), version='RGBRGB'))
# model = YOLO(model_cfg).load(pt_file)
model = YOLO(pt_file)


# 在验证的时候不会读取train_cfg
results = model.val(
    data=data_cfg,
    epochs=50, 
    device = [5],
    project = '/data3/litianhao/hsmot/yolo11',
    task = "obb",
    name = experiment_name,
    batch = 8,
    imgsz = 1280,
    cfg = train_cfg,#无用
    workers = 0,
    npy2rgb = True
    )