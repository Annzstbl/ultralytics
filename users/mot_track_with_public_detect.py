'''
Author: annzstbl@tianhaoli1996@gmail.com
Date: 2024-05-25 22:26:54
LastEditors: annzstbl@tianhaoli1996@gmail.com
LastEditTime: 2024-05-25 23:55:54
FilePath: /ultralytics/users/mot_track_wo_detect.py
Description: 

Copyright (c) 2024 by ${annzstbl}, All Rights Reserved. 
'''
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.engine.results import OBB
from ultralytics.engine.results import Results
import numpy as np
import os
import cv2
from tqdm import tqdm
from load_label import read_voc_xml_padding, load_yoloobb_txt_wo_conf

#这是和陈震香一致的类别
classes = ['car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle', 'awning-bike']

colors = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 红色
    (0, 0, 255),    # 蓝色
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 青色
    (128, 0, 0),    # 深红
    (0, 128, 0),    # 深绿
    (0, 0, 128),    # 深蓝
    (128, 128, 0),  # 榄
    (128, 0, 128),  # 紫红
    (0, 128, 128),  # 绿松石
]


img_path = '/data3/litianhao/datasets/nips2024/source_img/data23-20-80m-2146-1616-129-wu_00_394_463'
label_path = '/data/users/litianhao/ultralytics/users/debug_label_path'
save_img_path = '/data/users/litianhao/ultralytics/users/debug_save_img_path'
save_label_path = '/data/users/litianhao/ultralytics/users/debug_save_label_path'
tracker = '/data/users/litianhao/ultralytics/ultralytics/cfg/trackers/botsort.yaml'
cfg = IterableSimpleNamespace(**yaml_load(tracker))
tracker = BOTSORT(args=cfg, frame_rate=30)

save_txt = True
save_img = True

img_file_list = os.listdir(img_path)
img_file_list.sort()

for img_file in tqdm(img_file_list):
    img = cv2.imread(os.path.join(img_path, img_file))
    label_file = os.path.join(label_path, img_file.split('.')[0]+'.txt')
    assert os.path.exists(label_file), f'{label_file} not exists'
    if label_file.endswith('.xml'):
        box = read_voc_xml_padding(label_file)
    else:
        box = load_yoloobb_txt_wo_conf(label_file)

    # 把box从list转为ndarray
    box = np.array(box)
    result = OBB(box, orig_shape=img.shape[:2])
    tracks = tracker.update(result, img)
    # 把tracks写入文件
    if save_txt:
        with open(os.path.join(save_label_path, img_file.split('.')[0]+'.txt'), 'w') as f:
            for track in tracks:
                f.write(' '.join(map(str, track))+'\n')
    # 保存图片
    if save_img:
        # 画框 标注id
        for track in tracks:
            cx, cy, w, h, angle, track_id, _, cls_id, _ = track
            # 画旋转框
            color = colors[int(cls_id) % len(colors)]
            
            # 创建旋转矩形
            rect = ((cx, cy), (w, h), angle*180/np.pi)
            
            # 获取旋转矩形的4个顶点
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.polylines(img, [box_points], isClosed=True, color=color, thickness=2)
            
            # 标注id
            cv2.putText(img, str(int(track_id)), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imwrite(os.path.join(save_img_path, img_file), img)