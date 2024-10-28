from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
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

# read voc xml to rotate box
def read_voc_xml(xml_file):

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    root = tree.getroot()
    result = []
    for obj in root.findall('object'):
        robndbox = obj.find('robndbox')
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)
        # 分类
        cls_name = obj.find('name').text.replace('_human','')
        cls_id = classes.index(cls_name)
        result.append([cx, cy, w, h, angle, 1, cls_id])
    return result
        
        # read voc xml to rotate box
def read_voc_xml_padding(xml_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    root = tree.getroot()
    result = []
    for obj in root.findall('object'):
        if obj.find('robndbox') is None:
            continue
        robndbox = obj.find('robndbox')
        cx = float(robndbox.find('cx').text)-100
        cy = float(robndbox.find('cy').text)-100
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)
        # 分类
        cls_name = obj.find('name').text.replace('_human','')
        cls_id = classes.index(cls_name)
        result.append([cx, cy, w, h, angle, 1, cls_id])
    return result
    
if __name__ == '__main__':
    root_path = 'D:/lthWorkspace/Doctor/11workdir/nips2024/assign_data/11thanks_lx'
    img_root_path = os.path.join(root_path, 'img')
    label_path = os.path.join(root_path, 'labels_1')
    save_label_path = os.path.join(root_path, 'track_labels_1')
    save_img_root_path = os.path.join(root_path, 'track_img_1')

    save_img = True
    show_img = True
    read_voc_xml_func = read_voc_xml

    os.makedirs(save_label_path, exist_ok=True)


    img_path_list = os.listdir(img_root_path)
    img_path_list.sort()

    if show_img:

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1400, 1100)

    for img_path in img_path_list:

        tracker = 'ultralytics/cfg/trackers/botsort.yaml'
        # tracker = 'ultralytics/cfg/trackers/bytetrack.yaml'

        cfg = IterableSimpleNamespace(**yaml_load(tracker))
        tracker = BOTSORT(args=cfg, frame_rate=30)

        img_abs_path = os.path.join(img_root_path, img_path)
        save_img_path = os.path.join(save_img_root_path, img_path)
        os.makedirs(save_img_path, exist_ok=True)

        img_file_list = os.listdir(img_abs_path)
        img_file_list.sort()

        bar = tqdm(img_file_list)
        bar.set_description(f'Processing {img_path}')
        save_label_file = os.path.join(save_label_path, img_path +'.csv')
        if os.path.exists(save_label_file):
            os.remove(save_label_file)


        for img_file in bar:
            # 如果是json结尾
            if not img_file.endswith('.png'):
                continue
            save_img_file = os.path.join(save_img_root_path, img_path, img_file)
            label_file = os.path.join(label_path, img_file.split('.')[0]+'.xml')
            assert os.path.exists(label_file), f'{label_file} not exists'

            img = cv2.imread(os.path.join(img_abs_path, img_file))
            box = read_voc_xml_func(label_file)
            # 把box从list转为ndarray
            box = np.array(box)
            result = OBB(box, orig_shape=img.shape[:2])
            tracks = tracker.update(result, img)
            # 把tracks 和 reulst匹配起来，按照IOU, 最高的即为匹配结果
            for track in tracks:
                ind = track[8]
                track[:5] = result.xywhr[int(ind), :]

            with open(save_label_file, 'a') as f:
                frame_id = int(img_file.split('.')[0].split('_')[-1].split('-')[-1])
                neg =  -1
                # track : cx, cy, w, h, angle, track_id, score, cls_id, cls_score
                for track in tracks: 
                    xyxyxyxy = result.xyxyxyxy[int(track[8]), :]
                    cls = int(result.cls[int(track[8])])
                    track_id = int(track[5])
                    #写: frame_id, track_id, xyxyxyxy, neg, cls, neg
                    # xyxyxyxy用 [0][0] [0][1]索引
                    f.write(f'{frame_id},{track_id},{xyxyxyxy[0][0]}, {xyxyxyxy[0][1]}, {xyxyxyxy[1][0]}, {xyxyxyxy[1][1]}, {xyxyxyxy[2][0]}, {xyxyxyxy[2][1]}, {xyxyxyxy[3][0]}, {xyxyxyxy[3][1]}, {neg}, {cls}, {neg}\n')
                    
            # 保存图片
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
                # 标注idq
                cv2.putText(img, str(int(track_id)), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if show_img:
                cv2.imshow('img', img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            if save_img:
                cv2.imwrite(save_img_file, img)
    cv2.destroyAllWindows()    
