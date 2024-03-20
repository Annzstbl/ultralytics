import comet_ml
import os
from ultralytics import settings
from ultralytics import YOLO



# experiment_name = 'v8x_hod3k_rgb_pretrainedweight'
experiment_name = 'v8l_hod3k_hsi_fromscratch'
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
        project_name = "yolo8",
        experiment_name = experiment_name + '_' + comet_experiment_order,
    )


# Create a new YOLO model from scratch
# 不加载参数
# model = YOLO('yolov8n-16ch.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8l-16ch.yaml')
# model.load('/data3/litianhao/checkpoints/yolov8/yolov8l.pt')
# model = YOLO('/data3/litianhao/checkpoints/yolov8/yolov8x.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(
    data='/data3/litianhao/datasets/hod3k_hsi.yaml', 
    epochs=50, 
    device = [7],
    project = '/data3/litianhao/workdir/yolov8/detect/',
    name = experiment_name,
    batch = 16,
    imgsz = 640,
    hsv_h = 0, hsv_s = 0, hsv_v = 0,
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