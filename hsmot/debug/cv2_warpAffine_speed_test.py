import cv2
import numpy as np
import time

img = np.load('/data/users/litianhao/data/HSMOT/npy/data23-1/000002.npy')
# resize到1280
img_1280 = cv2.resize(img, (1280, 1280))
img_2560 = cv2.resize(img, (2560,2560))

print(f'img_1280 尺寸: {img_1280.shape}')
print(f'img_2560 尺寸: {img_2560.shape}')

img_1280_3ch = img_1280[:,:,:3]
img_2560_3ch = img_2560[:,:,:3]



# if self.perspective:
    # img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))

M = np.array([[    0.61035,           0,     -15.027],
       [          0,     0.61035,     -34.719],
       [          0,           0,           1]]) 

# 测试1000次所花费的时间

start = time.time()
for i in range(100):
    channel_list = cv2.split(img_1280)
    img_dst = cv2.merge([cv2.warpAffine(channel, M[:2], dsize=(640,640), borderValue=(114, 114, 114)) for channel in channel_list])
time_640_split = time.time() - start

start = time.time()
for i in range(100):
    img_dst = np.stack(
        [cv2.warpAffine(img_1280[:,:,channel], M[:2], dsize=(640,640), borderValue=(114, 114, 114)) for channel in range(img_1280.shape[2])], axis=2)
time_640_stack = time.time() - start

start = time.time()
for i in range(100):
    img_dst = cv2.warpAffine(img_1280_3ch, M[:2], dsize=(640,640), borderValue=(114, 114, 114))
time_640 = time.time() - start

start = time.time()
for i in range(100):
    channel_list = cv2.split(img_2560)
    img_dst = cv2.merge([cv2.warpAffine(channel, M[:2], dsize=(1280,1280), borderValue=(114, 114, 114)) for channel in channel_list])
time_1280_split = time.time() - start

start = time.time()
for i in range(100):
    img_dst = cv2.warpAffine(img_2560_3ch, M[:2], dsize=(1280,1280), borderValue=(114, 114, 114))
time_1280 = time.time() - start

print(f'img_1280 3ch warpAffine split: {time_640_split:.4f}s')
print(f'img_1280 3ch warpAffine: {time_640:.4f}s')
print(f'img_1280 3ch warpAffine stack: {time_640_stack:.4f}s')
print(f'img_2560 3ch warpAffine split: {time_1280_split:.4f}s')
print(f'img_2560 3ch warpAffine: {time_1280:.4f}s')
    