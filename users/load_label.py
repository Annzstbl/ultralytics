'''读取各种标注'''

import os
import numpy as np
import cv2

#这是和陈震香一致的类别
classes = ['car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle', 'awning-bike']


def load_det_dota_rotate4p_file(dota_file ,classes):
    '''
    
        读取 <检测器的结果> <dota格式> <rotate的4点类型> <文件>
        格式 [x1,y1 ..... x4, y4, cls_word, score]
        [719.0 212.0 750.0 215.0 744.0 279.0 714.0 277.0 car 0.94]
        转为 [cx, cy, w, h, angle, score, cls_id]

    '''
    labels = []
    with open(dota_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            x1, y1, x2, y2, x3, y3, x4, y4, cls_word, score = line
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, [x1, y1, x2, y2, x3, y3, x4, y4])
            score = float(score)
            cls_id = classes.index(cls_word)
            cx, cy, w, h, angle = vertices_to_rotated_rect(x1, y1, x2, y2, x3, y3, x4, y4)
            labels.append([cx, cy, w, h, angle, score, cls_id])



def vertices_to_rotated_rect(x1, y1, x2, y2, x3, y3, x4, y4):
    # 创建一个包含顶点坐标的数组
    vertices = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    
    # 使用 OpenCV 的 minAreaRect 函数计算最小外接矩形
    rect = cv2.minAreaRect(vertices)
    
    # 提取中心点坐标、尺寸和旋转角度
    (cx, cy), (w, h), angle = rect
    
    # OpenCV 的角度是从 -90 到 0 的，需要转换为 [0, 180] 范围
    if w < h:
        w, h = h, w
        angle += 90
    
    # 将角度转换为弧度
    angle_rad = angle * np.pi / 180
    
    return cx, cy, w, h, angle_rad




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
        cls_name = obj.find('name').text
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
    

def load_yoloobb_txt_wo_conf(txt_file):
    '''
    读取yoloobb的txt标注,不带conf
    '''
    result = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            cls_id, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line)
            cx, cy, w, h, angle = vertices_to_rotated_rect(x1, y1, x2, y2, x3, y3, x4, y4)
            result.append([cx, cy, w, h, angle, 1, cls_id])
    return result
