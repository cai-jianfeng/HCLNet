# _*_ coding:utf-8 _*_
"""
@Software: code
@FileName: confusion_matrix.py
@Date: 2023/10/15 16:04
@Author: caijianfeng
"""
from data_preprocess import Data
import os
from plot import plot_mode

label_path = "/path/to/label.xlsx"
label_num = 0
predict_folder = "/path/to"
com_CNN = "CNN_few.xlsx"
svm = "svm_few.xlsx"
PCLNet = "PCLNet_few.xlsx"
SSPRL = "SSPRL_few.xlsx"
HCLNet = "fine_few.xlsx"

patch_size = [15, 15]

data_util = Data()

# get predict label path
com_CNN_predict_path = os.path.join(predict_folder, com_CNN)
com_CNN_save_path = "/path/to/com_CNN_few.png"

svm_predict_path = os.path.join(predict_folder, svm)
svm_save_path = "/path/to/svm_few.png"

PCLNet_predict_path = os.path.join(predict_folder, PCLNet)
PCLNet_save_path = "./path/to/PCLNet_few.png"

SSPRL_predict_path = os.path.join(predict_folder, SSPRL)
SSPRL_save_path = "./path/to/SSPRL_few.png"

HCLNet_predict_path = os.path.join(predict_folder, HCLNet)
HCLNet_save_path = "./path/to/HCLNet_few.png"

# compute confusion matrix
plot_util = plot_mode()
if not os.path.exists(com_CNN_save_path):
    com_CNN_confusion_matrix = plot_util.plot_confusion_matrix(label_path=label_path,
                                                               predict_path=com_CNN_predict_path,
                                                               label_num=label_num,
                                                               patch_size=patch_size,
                                                               save_path=com_CNN_save_path)
if not os.path.exists(svm_save_path):
    svm_confusion_matrix = plot_util.plot_confusion_matrix(label_path=label_path,
                                                           predict_path=svm_predict_path,
                                                           label_num=label_num,
                                                           patch_size=patch_size,
                                                           save_path=svm_save_path)
if not os.path.exists(SSPRL_save_path):
    SSPRL_confusion_matrix = plot_util.plot_confusion_matrix(label_path=label_path,
                                                             predict_path=SSPRL_predict_path,
                                                             label_num=label_num,
                                                             patch_size=patch_size,
                                                             save_path=SSPRL_save_path)
if not os.path.exists(PCLNet_save_path):
    PCLNet_confusion_matrix = plot_util.plot_confusion_matrix(label_path=label_path,
                                                              predict_path=PCLNet_predict_path,
                                                              label_num=label_num,
                                                              patch_size=patch_size,
                                                              save_path=PCLNet_save_path)
if not os.path.exists(HCLNet_save_path):
    HCLNet_confusion_matrix = plot_util.plot_confusion_matrix(label_path=label_path,
                                                              predict_path=HCLNet_predict_path,
                                                              label_num=label_num,
                                                              patch_size=patch_size,
                                                              save_path=HCLNet_save_path)