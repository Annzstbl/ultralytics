'''
Author: annzstbl@tianhaoli1996@gmail.com
Date: 2024-05-22 23:00:07
LastEditors: annzstbl@tianhaoli1996@gmail.com
LastEditTime: 2024-05-25 22:37:34
FilePath: /ultralytics/users/mot_track.py
Description: 

Copyright (c) 2024 by ${annzstbl}, All Rights Reserved. 
'''
import os
import cv2 
from ultralytics import YOLO
import numpy as np
import torch
import tqdm
from collections import defaultdict



colors = [
    (0, 255, 0),    # ignored regions - 绿色
    (255, 0, 0),    # pedestrian - 红色
    (0, 0, 255),    # people - 蓝色
    (255, 255, 0),  # bicycle - 黄色
    (255, 0, 255),  # car - 紫色
    (0, 255, 255),  # van - 青色
    (128, 0, 0),    # truck - 深红
    (0, 128, 0),    # tricycle - 深绿
    (0, 0, 128),    # awning-tricycle - 深蓝
    (128, 128, 0),  # bus - 橄榄
    (128, 0, 128),  # motor - 紫红
    (0, 128, 128),  # others - 绿松石
]

videos_path = '/data3/litianhao/datasets/nips2024/MOT/videos/'
imgs_save_path = '/data3/litianhao/datasets/nips2024/MOT/tracking_result'
save_img = False

for video_path in tqdm.tqdm(os.listdir(videos_path)):
    video_path = os.path.join (videos_path, video_path)
    img_save_path = os.path.join(imgs_save_path, video_path.split('/')[-1].split('.')[0])
    if os.path.exists(img_save_path) and save_img == True:
        # 跳过已经生成过的
        continue
    os.makedirs(img_save_path, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    model = YOLO('/data/users/litianhao/ultralytics/project/mot/nips_v8lobb_pretrainedweight4/weights/last.pt')

    track_history = defaultdict(lambda: [])

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break

        results = model.track(frame,persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else ['N' * len(boxes)]

        annotated_frame = results[0].plot()
                # Plot the tracks
        
        for box, track_id, cls in zip(boxes, track_ids, results[0].boxes.cls):
            x, y, w, h = box
            track = track_history[track_id]

            #画框，写id
            annotated_frame = cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), colors[int(cls)], 2)
            annotated_frame = cv2.putText(frame, str(track_id), (int(x - w / 2), int(y - h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[int(cls)], 2)


            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # save img
        if save_img:
            cv2.imwrite(f'{img_save_path}/{i:06d}.jpg', annotated_frame)
        i += 1




    