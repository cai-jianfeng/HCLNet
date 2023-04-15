"""
created on:2022/11/13 11:29
@author:caijianfeng
"""
import torch
from torch import nn
import torch.nn.functional as F


class HetNet(nn.Module):
    def __init__(self, T, device=None):
        super(HetNet, self).__init__()
        self.dim = 30
        self.T = T
        self.device = device
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=(3,),
                                  padding=(2,))
        self.batchnorm1d_1 = nn.BatchNorm1d(num_features=3)
        self.conv1d_2 = nn.Conv1d(in_channels=3,
                                  out_channels=9,
                                  kernel_size=(3,),
                                  padding=(2,))
        self.batchnorm1d_2 = nn.BatchNorm1d(num_features=9)
        self.conv2d_1 = nn.Conv2d(in_channels=18,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  padding=(2, 2))
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=32)
        self.fc = nn.Linear(45, 30)
        self.conv2d_2 = nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  padding=(1, 1))
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=64)
        self.mp_1d = nn.MaxPool1d(kernel_size=(2,))
        self.mp_2d = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2 * 30)
        # self.batchnorm1d_3 = nn.BatchNorm1d(num_features=60)
        self.fc4 = nn.Linear(2 * 30, 30)
        self.dropout = nn.Dropout()
    
    def forward(self, data1, data2):
        in_size1 = data1.size(0)  # data1_size[0] = batch size
        x1 = self.mp_1d(F.relu(self.batchnorm1d_1(self.conv1d_1(data1))))
        x1 = self.mp_1d(F.relu(self.batchnorm1d_2(self.conv1d_2(x1))))
        x1 = x1.view(in_size1, -1)
        x1 = self.dropout(self.fc(x1))
        x1 = nn.functional.normalize(x1, dim=1)
        in_size2 = data2.size(0)  # data2_size[0] = batch size
        x2 = self.mp_2d(F.relu(self.batchnorm2d_1(self.conv2d_1(data2))))
        x2 = self.mp_2d(F.relu(self.batchnorm2d_2(self.conv2d_2(x2))))
        x2 = x2.view(in_size2, -1)
        # x2 = F.relu(self.batchnorm1d_3(self.fc2(x2)))
        x2 = self.dropout(F.relu(self.fc2(x2)))
        x2 = self.dropout(self.fc3(x2))
        x2 = self.dropout(self.fc4(x2))
        x2 = nn.functional.normalize(x2, dim=1)
        # TODO:queue selection (each x2 writes every other x1 (in a batch) except the corresponding position to the queue as a negative sample)
        # x_pos = torch.einsum('nc, nc -> n', [x2, x1]).unsqueeze(-1)
        # queue = torch.randn(self.dim, in_size1 - 1)
        # x1 = x1.T  # (dim, batch_size)
        # print(x1.shape)
        # x_neg = torch.zeros((in_size1, in_size1 - 1)).to(self.device)  # (batch_size, batch_size-1)
        # for i, x in enumerate(x2):
        #     queue = torch.concat((x1[:, :i], x1[:, i+1:]), dim=1)  # (dim, batch_size-1)
        #     x_neg[i] = torch.einsum('c, cm -> m', [x, queue])  # m = n-1
        #
        # # x_neg = torch.einsum('nc, c(n-1) -> n(n-1)', [x2, queue])
        # # print(x_pos.shape, x_neg.shape)
        # logits = torch.concat([x_pos, x_neg], dim=1)
        # print(x1.shape, x1.T.shape)
        # print(x2.shape)
        logits = torch.einsum('nc, cm -> nm', [x2, x1.T])  # (batch_size, batch_size) m=n
        logits /= self.T
        labels = torch.tensor([i for i in range(in_size1)], dtype=torch.long).to(self.device)
        # labels = torch.tensor([i for i in range(in_size1)], dtype=torch.long)
        return logits, labels

class Class_Net(nn.Module):
    def __init__(self, num_class=4):
        super(Class_Net, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=18,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  padding=(2, 2))
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=32)
        self.conv2d_2 = nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  padding=(1, 1))
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=64)
        self.mp_1d = nn.MaxPool1d(kernel_size=(2,))
        self.mp_2d = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc2 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_class)
        # self.dropout = nn.Dropout()

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp_2d(F.relu(self.conv2d_1(x)))
        x = self.mp_2d(F.relu(self.conv2d_2(x)))
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.dropout(self.fc(x))
        return x


class DClass_Net(nn.Module):
    def __init__(self, num_class=4):
        super(DClass_Net, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, kernel_size=3, padding=2, dtype=torch.float32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=2, dtype=torch.float32)
        self.conv3 = nn.Conv2d(9, 16, kernel_size=3, padding=2, dtype=torch.float32)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=2, dtype=torch.float32)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        in_size = x.size(0)
        x1 = self.mp(F.relu(self.conv1(x[:, 0:9])))
        x1 = self.mp(F.relu(self.conv2(x1)))
        x2 = self.mp(F.relu(self.conv3(x[:, 9:18])))
        x2 = self.mp(F.relu(self.conv4(x2)))
        x = torch.cat((x1.view(in_size, -1), x2.view(in_size, -1)), 1)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        return x

class Beam_search_Net(nn.Module):
    def __init__(self, num_class=4, features_num = 28):
        super(Beam_search_Net, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=features_num,
                                  out_channels=32,
                                  kernel_size=(3,),
                                  padding=(2,))
        self.conv1d_2 = nn.Conv1d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(3,),
                                  padding=(2,))
        self.mp_1d = nn.MaxPool1d(kernel_size=(2,))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1d_1(x)))
        x = self.mp(F.relu(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        return x
