'''
Author: annzstbl@tianhaoli1996@gmail.com
Date: 2024-05-23 22:40:13
LastEditors: annzstbl@tianhaoli1996@gmail.com
LastEditTime: 2024-05-24 20:04:48
FilePath: /ultralytics/users/mot_detect.py
Description: 

Copyright (c) 2024 by ${annzstbl}, All Rights Reserved. 
'''
from ultralytics import YOLO
import os

model = YOLO('/data/users/litianhao/ultralytics/project/mot/nips_v8lobb_pretrainedweight4/weights/last.pt')
# model = YOLO('/data3/litianhao/checkpoints/yolov8/yolov8l-obb.pt')

img_paths = '/data3/litianhao/datasets/nips2024/source_img/'
# predict_path = '/data3/litianhao/datasets/nips2024/source_img_predict/'
# os.makedirs(predict_path, exist_ok=True)

# grab_img_paths = '/data3/litianhao/datasets/nips2024/source_img/*/*.png'
# results = model(grab_img_paths, save_txt = True, device = [0], imgsz=(1280,1280))
img_paths_it = os.listdir(img_paths)
img_paths_it.sort()

for img_path in img_paths_it:

    # img_path 名字是dataxx-..提取xx作为数字, 过滤处理大于37小于44的
    # if int(img_path[4:6]) > 37 and int(img_path[4:6]) < 44:
    if int(img_path[4:6]) >= 44:
        img_it = os.listdir(os.path.join(img_paths, img_path))
        img_it.sort()
        for img in img_it:
            img = os.path.join(img_paths, img_path, img)
            results = model(img, save_txt = True, device = [0], imgsz=(1280,1280))

