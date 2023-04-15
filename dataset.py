"""
created on:2023/2/14 20:05
@author:caijianfeng
@Purpose: PolSAR_Dataset is load data for CL, fine-tuning/train, and eval of our method/compared method
          features_Dataset is load data for training classifier of Beam Search
"""
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from data_preprocess import Data


class PolSAR_Dataset(Dataset):
    def __init__(self, dataset_path, mode='contrastive', transform=None):
        """
        Read Data
        :param dataset_path: Dataset path
        :param mode: Model pattern(contrastive -> CL
                             train/fine -> transfer learning(Downstream tasks)
                             eval -> test(Downstream tasks))
        :param transform: tensor transform
        """
        super(PolSAR_Dataset, self).__init__()
        # TODO:Write data to read directly, split in Dataset (no need to store and read)
        self.data = Data()
        self.dataset_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.data_paths = None
        if self.mode == 'contrastive':
            self.data_T_paths = []
            self.data_F_paths = []
            with(open(os.path.join(self.dataset_path, 'contrastive.txt'), 'r')) as f:
                self.data_infos = f.readlines()
            for data_info in self.data_infos:
                data_T_path, data_F_path = data_info.strip().split(';')
                self.data_T_paths.append(data_T_path)
                self.data_F_paths.append(data_F_path)
        elif self.mode in ['train', 'fine']:
            self.data_paths = []
            self.labels = []
            with(open(os.path.join(self.dataset_path, self.mode+'.txt'), 'r')) as f:
                self.data_infos = f.readlines()
            for data_info in self.data_infos:
                data_T_path, label = data_info.strip().split('\t')
                self.data_paths.append(data_T_path)
                self.labels.append(int(label))
        elif self.mode == 'train_superpixel':
            self.data_paths = []
            self.labels = []
            with(open(os.path.join(self.dataset_path, 'train_superpixel.txt'), 'r')) as f:
                self.data_infos = f.readlines()
                for data_info in self.data_infos:
                    data_T_path, label = data_info.strip().split('\t')
                    self.data_paths.append(data_T_path)
                    self.labels.append(int(label))
        else:
            self.data_paths = []
            self.labels = []
            with(open(os.path.join(self.dataset_path, 'eval.txt'), 'r')) as f:
                self.data_infos = f.readlines()
                for data_info in self.data_infos:
                    data_T_path, label = data_info.strip().split('\t')
                    self.data_paths.append(data_T_path)
                    self.labels.append(int(label))
    
    def __getitem__(self, index):
        if self.mode == 'contrastive':
            data_T_path = self.data_T_paths[index]
            data_F_path = self.data_F_paths[index]
            data_T = self.data.get_data_list(data_path=data_T_path)
            data_T = self.data.data_dim_change(data=data_T)
            data_T = np.array(data_T).astype('float32')
            data_T = self.transform(data_T)
            data_F = self.data.get_feature_data_by_file_28(feature_data_path=data_F_path, sheet_num=0)
            data_F = np.array(data_F).astype('float32')
            # data_F = self.transform(data_F)
            data_F = torch.tensor(data_F).unsqueeze(dim=0)
            return data_T, data_F
        
        else:
            data_path = self.data_paths[index]
            data = self.data.get_data_list(data_path=data_path)
            data = self.data.data_dim_change(data)
            data = np.array(data).astype('float32')
            data = self.transform(data)
            label = self.labels[index]
            label = torch.tensor(label, dtype=torch.int64)
            return data, label
    
    def __len__(self):
        return len(self.data_paths) if self.data_paths is not None else len(self.data_T_paths)


class features_Dataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Read Data
        :param dataset_path: Dataset path
        :param transform: tensor transform
        """
        super(features_Dataset, self).__init__()
        self.data = Data()
        self.dataset_path = dataset_path
        self.transform = transform
        self.data_paths = None
        self.data_paths = []
        self.labels = []
        with(open(os.path.join(self.dataset_path, 'bs_train.txt'), 'r')) as f:
            self.data_infos = f.readlines()
        for data_info in self.data_infos:
            data_path, label = data_info.strip().split('\t')
            self.data_paths.append(data_path)
            self.labels.append(int(label))

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data = self.data.get_data_list(data_path=data_path)
        data = self.data.data_dim_change(data)
        data = np.array(data).astype('float32')
        data = self.transform(data)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.int64)
        return data, label

    def __len__(self):
        return len(self.data_paths)