# 车辆统计
# 

from ultralytics import YOLO
import os.path as osp
import os
import cv2
from collections import defaultdict
from tqdm import tqdm
import sys

'''
2 car
5 bus
7 truck

'''
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

# Load an official or custom model
model = YOLO("yolov8l.pt")  # Load an official Detect model
# model = YOLO("yolov8n-seg.pt")  # Load an official Segment model
# model = YOLO("yolov8n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model



def run(video_name , prefix):
    video_root_path = '/data/users/litianhao/data/video'
    img_root_path = '/data/users/litianhao/data/results/imgs'
    save_video_root_path = '/data/users/litianhao/data/results/video'
    txt_root_path = '/data/users/litianhao/data/results/txt'
    # video_root_path = '/data3/PublicDataset/Custom/traffic_counter'
    # img_root_path = '/data3/PublicDataset/Custom/traffic_counter/results/imgs'
    # save_video_root_path = '/data3/PublicDataset/Custom/traffic_counter/results/video'
    # txt_root_path = '/data3/PublicDataset/Custom/traffic_counter/results/txt'

    video1 = osp.join(video_root_path, prefix, video_name)
    video_simple_name = video_name.split('.')[0]
    img_save_path = osp.join(img_root_path, video_simple_name)
    txt_save_file = osp.join(txt_root_path, f'{video_simple_name}.txt')
    save_video_file = osp.join(save_video_root_path, f'{video_simple_name}.avi')
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(txt_root_path, exist_ok=True)
    os.makedirs(save_video_root_path, exist_ok=True)

    # 清空txt
    with open(txt_save_file, 'w') as f:
        pass

    cap = cv2.VideoCapture(video1)

    i=0 
    save_img = True
    save_txt = True

    # 一共有多少帧
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bar = tqdm(total=frame_count)

    while True:

        # 更新bar
        bar.update(1)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break

        results = model.track(frame,persist=True, tracker = 'car_counter.yaml')


        if results[0].boxes.id is None or results[0].boxes.conf is None:
            i += 1
            continue

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else ['N' * len(boxes)]
        confs = results[0].boxes.conf.cpu().tolist() if results[0].boxes.conf is not None else ['N' * len(boxes)]
        # annotated_frame = results[0].plot()
                # Plot the tracks
        

        result = []

        for box, track_id, conf, cls in zip(boxes, track_ids, confs, results[0].boxes.cls):
            if cls != 2 and cls != 5 and cls != 7:
                continue
            cx, cy, w, h = box
            result.append((i,track_id, int(cx), int(cy), int(w), int(h), int(cls), conf))

        # save result
        if save_txt:
            with open(txt_save_file, 'a') as f:
                for track in result:
                    f.write(' '.join(map(str, track))+'\n')

        # save img
        if save_img:
            for track in result:
                _, track_id, cx, cy, w, h, cls, conf = track

                color = colors[int(cls)]
                cv2.rectangle(frame, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2),), color, 2)
                # 标注id
                cv2.putText(frame, str(track_id), (int(cx - w / 2), int(cy - h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(f'{img_save_path}/{i:06d}.jpg', frame)
        i += 1
        # if i == 100:
        #     break
    
    print('generate video')
    # 所有图像转视频
    img_files = os.listdir(img_save_path)
    img_files.sort()
    img = cv2.imread(os.path.join(img_save_path, img_files[0]))
    height, width, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(save_video_file, fourcc, 30, (width, height))
    for img_file in tqdm(img_files, desc='generate video'):
        img = cv2.imread(os.path.join(img_save_path, img_file))
        video.write(img)
    video.release()
    bar.close()

if __name__ == '__main__':

    assert len(sys.argv) == 3, 'Usage: python car_counter.py video_name prefix'

    video_name = sys.argv[1]
    prefix = sys.argv[2]

    print('video_name:', video_name)
    print('prefix:', prefix)

    run(video_name, prefix)