'''
Author: annzstbl@tianhaoli1996@gmail.com
Date: 2024-03-15 10:57:05
LastEditors: annzstbl@tianhaoli1996@gmail.com
LastEditTime: 2024-05-23 22:33:54
FilePath: /ultralytics/users/main_train.py
Description: 

Copyright (c) 2024 by ${annzstbl}, All Rights Reserved. 
'''
import comet_ml
import os
from ultralytics import settings
from ultralytics import YOLO



# experiment_name = 'v8x_hod3k_rgb_pretrainedweight'
experiment_name = 'nips_v8lobb_pretrainedweight'
comet_experiment_order = '1'
comet_enable = True

if comet_enable:
    # proxy for upload comet
    os.environ['HTTP_PROXY'] = '10.106.14.29:20811'
    os.environ['HTTPS_PROXY'] = '10.106.14.29:20811'
    # comet setting
    os.environ['COMET_API_KEY'] = 'rhueIeiHA5nooGr4p2jgOkah6'
        #level: workspace-> project_name -> experiment_name
    comet_ml.init(
        project_name = "nips2024",
        experiment_name = experiment_name + '_' + comet_experiment_order,
    )


# Create a new YOLO model from scratch
# 不加载参数
# model = YOLO('yolov8n-16ch.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8l-16.yaml')
# model.load()
# model.load('/data3/litianhao/checkpoints/yolov8/yolov8l.pt')
model = YOLO('/data3/litianhao/checkpoints/yolov8/yolov8l-obb.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(
    data='/data3/litianhao/datasets/nips2024/nips2024.yaml', 
    epochs=50, 
    device = [0, 1],
    project = '/data3/litianhao/workdir/yolov8/mot/',
    name = experiment_name,
    batch = 16,
    imgsz = 1280,
    # hsv_h = 0, hsv_s = 0, hsv_v = 0,
    # evolve = True,
    )

# Evaluate the model's performance on the validation set
# results = model.val()

# Export the model to ONNX format
# success = model.export(format='onnx')

# Debug
# 创建一个16维度的输入矩阵
# import torch
# test_input_image = torch.rand(1, 16, 640, 640)
# result = model(test_input_image)
# print(result.shape)