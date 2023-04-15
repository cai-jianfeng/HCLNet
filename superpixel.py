# _*_ coding:utf-8 _*_
"""
@Software: code
@FileName: superpixel.py
@Date: 2022/12/12 9:18
@Author: caijianfeng
@Purpose: This file converts the .mat superpixel segmentation results into a visualization(.png)
          and selects a training set based on the superpixel segmentation
"""
import scipy.io as scio
import numpy as np
import cv2
import xlrd
from tqdm import tqdm
import json
import random
from data_preprocess import Data

# In[0] read superpixel
file_path = '/path/to/superpixel.mat'
matdata = scio.loadmat(file_path)
# print(matdata.keys())
useful_label = matdata['useful_sp_label']
nums = []
patch_size = [15, 15]
for i in tqdm(range(useful_label.shape[0])):
    for j in range(useful_label.shape[1]):
        if useful_label[i][j] not in nums:
            nums.append(useful_label[i][j])
print(len(nums))  # 2719

# In[1] plot superpixel
color_path = '/path/to/color.xlsx'
colors = []
color_sheets = xlrd.open_workbook(color_path)
color_sheet = color_sheets.sheet_by_index(0)
rows = color_sheet.nrows
cols = color_sheet.ncols
for i in range(rows):
    color = [color_sheet.cell_value(i, 0), color_sheet.cell_value(i, 1), color_sheet.cell_value(i, 2)]
    colors.append(color)
print('color mat load succeed!')
R = np.zeros_like(useful_label, dtype='uint8')
G = np.zeros_like(useful_label, dtype='uint8')
B = np.zeros_like(useful_label, dtype='uint8')
label_file = '/path/to/label.xlsx'
label_book = xlrd.open_workbook(label_file)
label_sheet = label_book.sheet_by_index(0)
for i in tqdm(range(useful_label.shape[0])):
    for j in range(useful_label.shape[1]):
        label = int(label_sheet.cell_value(i, j))
        R[i][j] = colors[label][0]
        G[i][j] = colors[label][1]
        B[i][j] = colors[label][2]
for i in tqdm(range(useful_label.shape[0])):
    for j in range(useful_label.shape[1] - 1):
        if useful_label[i][j] != useful_label[i][j + 1]:
            R[i][j] = R[i][j + 1] = 255
            G[i][j] = G[i][j + 1] = 255
            B[i][j] = B[i][j + 1] = 255

for j in tqdm(range(useful_label.shape[1])):
    for i in range(useful_label.shape[0] - 1):
        if useful_label[i][j] != useful_label[i + 1][j]:
            R[i][j] = R[i + 1][j] = 120
            G[i][j] = G[i + 1][j] = 120
            B[i][j] = B[i + 1][j] = 120

superpixel_map = cv2.merge([B, G, R])
# cv2.imshow('label', superpixel_map)
# cv2.waitKey(0)
save_path = '/path/to/superpixel.png'
cv2.imwrite(save_path, superpixel_map)

# In[2] cluster each superpixel
superpixel_dict = dict()
for i in tqdm(range(useful_label.shape[0] - patch_size[0])):
    for j in range(useful_label.shape[1] - patch_size[1]):
        superpixel_dict.setdefault(int(useful_label[i + patch_size[0] // 2][j + patch_size[1] // 2]), [])
        superpixel_dict[int(useful_label[i + patch_size[0] // 2][j + patch_size[1] // 2])].append([i, j])

# In[3] save as json
superpixel_json = json.dumps(superpixel_dict)
with open('/path/to/superpixel.json', 'w', encoding='utf-8') as f:
    json.dump(superpixel_json, f, ensure_ascii=False, indent=2)

# In[4] save train data_flevoland4
fpath = '/path/to/superpixel.json'
train_superpixel_list_path = '/path/to/train_superpixel.txt'
num_superpixels = 0
data_size = num_superpixels / 100  # num_superpixels / 100
batch_size = 64
dim = []  # the whole map size(subtract the patch size)
data_util = Data()
label_path = '/path/to/label.xlsx'
label_set = data_util.get_label_list(label_path=label_path)
with open(fpath, 'r') as f:
    d = json.load(f)
d = json.loads(d)
superpixel_ids = random.sample(range(len(d.keys())), data_size)
# print(superpixel_id)
train_superpixel_list = []
for superpixel_id in superpixel_ids:
    datas = d[str(superpixel_id)]
    data_ids = [i for i in range(len(datas))]
    random.shuffle(data_ids)
    i = 0
    for data_id in data_ids:
        data_path = '/path/to/TRI/TRI' + str(datas[data_id][0] * dim[1] + datas[data_id][1]) + '.xlsx'
        label = label_set[datas[data_id][0] + patch_size[0] // 2][datas[data_id][1] + patch_size[1] // 2]
        if int(label) != 0:
            train_superpixel_list.append(data_path + '\t%d' % label + '\n')
            i += 1
        if i >= batch_size:
            break
    if i < batch_size:
        print(superpixel_id, ';', len(datas), '; ', i)

with open(train_superpixel_list_path, 'a') as f:  # load superpixel train dataset info
    for train_superpixel_data in train_superpixel_list:
        f.write(train_superpixel_data)
print('save succeed!')
