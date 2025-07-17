import os
import numpy as np
from glob import glob
import shutil

# 原始 .npy 数据目录
NPY_ROOT = "/data/users/litianhao/data/hsmot/test/npy"
# 所有预测图像的输出目录（没有按vid分类）
PREDICT_FLAT_DIR = "/data3/litianhao/hsmot/paper/yolo118ch"
PREDICT_FLAT_DIR_NEW = "/data3/litianhao/hsmot/paper/yolo118ch_new"
os.makedirs(PREDICT_FLAT_DIR_NEW, exist_ok=True)



# Step 1: 获取有序的所有 npy 路径和对应目标文件名
target_names = []
src_names = []
vid_list = sorted(os.listdir(NPY_ROOT))
for vid in vid_list:
    vid_dir = os.path.join(NPY_ROOT, vid)
    npy_files = sorted(os.listdir(vid_dir))  # e.g., ['000001.npy', '000002.npy', ...]
    for npy_file in npy_files:
        frame = os.path.splitext(npy_file)[0]
        src_names.append(f"{vid}_{frame}.png")

for vid in os.listdir(NPY_ROOT):
    vid_dir = os.path.join(NPY_ROOT, vid)
    npy_files = sorted(os.listdir(vid_dir))  # e.g., ['000001.npy', '000002.npy', ...]
    for npy_file in npy_files:
        frame = os.path.splitext(npy_file)[0]
        target_names.append(f"{vid}_{frame}.png")


# Step 2: 获取预测输出图像
# pred_imgs = sorted(glob(os.path.join(PREDICT_FLAT_DIR, '*.jpg')) + glob(os.path.join(PREDICT_FLAT_DIR, '*.png')))[1:]
if len(src_names) != len(target_names):
    print(f"[Warning] #npy = {len(target_names)}, #pred_imgs = {len(pred_imgs)}. Will rename min(len).")
min_len = min(len(target_names), len(src_names))

# Step 3: 按顺序重命名
for i in range(min_len):
    old_path = os.path.join(PREDICT_FLAT_DIR, src_names[i])
    ext = os.path.splitext(old_path)[1]  # 保留原扩展名（.png or .jpg）
    new_path = os.path.join(PREDICT_FLAT_DIR_NEW, os.path.splitext(target_names[i])[0] + ext)
    shutil.move(old_path, new_path)

print(f"[Done] Renamed {min_len} predicted images to vid_frame.png format.")
