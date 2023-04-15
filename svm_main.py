# _*_ coding:utf-8 _*_
"""
@Software: code
@FileName: svm_main.py
@Date: 2022/12/15 14:48
@Author: caijianfeng
"""
from sklearn import svm
import os
from data_preprocess import Data
import numpy as np
# from sklearn.externals import joblib
import joblib
from tqdm import tqdm
import xlsxwriter
import json

# In[0] generate train dataset
contrastive_path = '/path/to/contrastive.txt'
train_list_path = '/path/to/train.txt'
eval_list_path = '/path/to/eval.txt'
predict_list_path = '/path/to/predict.txt'
patch_size = [15, 15]  # in this article is [15, 15]; you can change it
data_path = ['/path/to/TR.xlsx', '/path/to/TI.xlsx']
label_path = '/path/to/label.xlsx'
target_path = '/path/to/'
if not os.path.exists(train_list_path):
    print('------data generate------')
    data_util = Data()
    data_util.save_data_label_segmentation_TRI(data_path=data_path,
                                               label_path=label_path,
                                               target_path=target_path,
                                               contrastive_path=contrastive_path,
                                               train_list_path=train_list_path,
                                               eval_list_path=eval_list_path,
                                               patch_size=patch_size,
                                               predict_list_path=predict_list_path)
print('------data generate succeed!------')

# In[1] 读取数据集(train)
dataset_path = '/path/to/'
train_data_paths = []
train_labels = []
with open(os.path.join(dataset_path, 'train.txt'), 'r', encoding='utf-8') as f:
    info = f.readlines()

for data_info in info:
    data_T_path, label = data_info.strip().split('\t')
    train_data_paths.append(data_T_path)
    train_labels.append(label)
print('train data path read succeed!')

# In[2] read train dataset and train
data_util = Data()
train_datas = []
for data_path in tqdm(train_data_paths):
    train_datas.append(data_util.get_data_list(data_path=data_path))
train_datas = np.array(train_datas)
train_datas = train_datas.reshape((train_datas.shape[0], -1))
print(train_datas.shape)

svm_model = svm.SVC(kernel='rbf')
svm_model.fit(train_datas, train_labels)

joblib.dump(svm_model, '/path/to/svm.pkl')
train_score = svm_model.score(train_datas, train_labels)
print('train acc:', train_score)

# In[3] read eval dataset and eval
eval_data_paths = []
eval_labels = []
with open(os.path.join(dataset_path, 'eval.txt'), 'r', encoding='utf-8') as f:
    info = f.readlines()
for data_info in info:
    data_T_path, label = data_info.strip().split('\t')
    eval_data_paths.append(data_T_path)
    eval_labels.append(label)
svm_model = joblib.load('/path/to/svm.pkl')
eval_datas = []
for data_path in tqdm(eval_data_paths):
    eval_datas.append(data_util.get_data_list(data_path=data_path))
eval_datas = np.array(eval_datas)
eval_datas = eval_datas.reshape((eval_datas.shape[0], -1))
print(eval_datas.shape)

eval_score = svm_model.score(eval_datas, eval_labels)
print('eval acc:', eval_score)

# In[4] read predict dataset and predict
predict_data_paths = []
predict_labels = []
readme_json = '/path/to/TRI/readme.json'
with open(readme_json, 'r') as f:
    data_info = json.load(f)
dim = data_info['dim']
with open(os.path.join(dataset_path, 'predict.txt'), 'r', encoding='utf-8') as f:
    info = f.readlines()
for data_info in info:
    data_T_path, label = data_info.strip().split('\t')
    predict_data_paths.append(data_T_path)
    predict_labels.append(label)
print('data path load succeed!')
print(len(predict_data_paths) == dim[0] * dim[1])
predict_book = xlsxwriter.Workbook('/path/to/predict_labels_svm.xlsx')
predict_sheet = predict_book.add_worksheet('sheet')
# predict_datas = []
row, col = 0, 0
for data_path in tqdm(predict_data_paths):
    predict_data = data_util.get_data_list(data_path=data_path)
    predict_label = svm_model.predict(predict_data.reshape((1, -1)))
    predict_sheet.write(row, col, predict_label[0])
predict_book.close()
print('predict end!')
