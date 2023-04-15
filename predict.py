# _*_ coding:utf-8 _*_
"""
@Software: flevoland15_code
@FileName: predict.py
@Date: 2023/4/11 17:57
@Author: caijianfeng
"""

import torch
import numpy as np
import xlsxwriter

from data_preprocess import Data
from network import DClass_Net
from tqdm import tqdm
import os

patch_size = [15, 15]
data_path = ['/path/to/T_R.xlsx', '/path/to/T_I.xlsx']
label_path = '/path/to/label.xlsx'
data = Data()
data_R_set = data.get_data_list(data_path=data_path[0])
data_T_set = data.get_data_list(data_path=data_path[1])
dim = data_R_set.shape
cnet = DClass_Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
cnet_param = torch.load('/path/to/fine_tuning.pkl', map_location=device)
cnet.load_state_dict(cnet_param)
cnet.to(device=device)

target_path = '/path/to/fine_tuning.xlsx'
target_path = os.path.join(target_path)
book = xlsxwriter.Workbook(filename=target_path)
sheet = book.add_worksheet('sheet')
rs = tqdm(range(0, dim[1] - patch_size[0]))
for i in rs:
    for j in range(0, dim[2] - patch_size[1]):
        data_R = data_R_set[:, i:i + patch_size[0], j:j + patch_size[1]]
        data_T = data_T_set[:, i:i + patch_size[0], j:j + patch_size[1]]
        predict_data = np.concatenate((data_R, data_T), axis=0)
        predict_data = np.expand_dims(predict_data, 0)
        predict_data = torch.tensor(predict_data, dtype=torch.float32)
        predict_data = predict_data.to(device)
        predict_label = cnet(predict_data)
        _, predict_label = torch.max(predict_label.data, dim=1)
        sheet.write(i, j, predict_label.item())
        rs.desc = 'predict -> {}/{}'.format(i, j)
book.close()
