# _*_ coding:utf-8 _*_
"""
@Software: code
@FileName: fine_tuning.py
@Date: 2022/12/3 15:11
@Author: caijianfeng
"""
import torch
import xlsxwriter
from torch import nn
import torch.optim as optim
from torchvision import transforms
import os
from network import Class_Net, HetNet
from data_preprocess import Data
from dataset import PolSAR_Dataset
from torch.utils.data import DataLoader
from net_run import run
import xlsxwriter


# In[1] generate train dataset
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

# In[2] data_flevoland15 load(train and eval)
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, ), (0.229, ))
])
dataset_path = '/path/to'
train_dataset = PolSAR_Dataset(dataset_path=dataset_path,
                               mode='fine',
                               transform=transform)
eval_dataset = PolSAR_Dataset(dataset_path=dataset_path,
                              mode='eval',
                              transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
eval_loader = DataLoader(eval_dataset,
                         shuffle=False,
                         batch_size=batch_size)

# In[3] generate network via pretrained network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnet = Class_Net()
hnet = HetNet(T=0.07)
contrastive_para = torch.load('/path/to/pretrained.pkl')
hnet.load_state_dict(contrastive_para)
cnet.conv2d_1 = hnet.conv2d_1
cnet.conv2d_2 = hnet.conv2d_2
cnet.batchnorm2d_1 = hnet.batchnorm2d_1
cnet.batchnorm2d_2 = hnet.batchnorm2d_2
cnet.mp_2d = hnet.mp_2d
cnet.dropout = hnet.dropout

cnet = cnet.to(device)
print(cnet)

# In[4] loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.0005
optimizer = optim.Adam(params=cnet.parameters(),
                       lr=lr,
                       weight_decay=1e-4)

# In[5] train
run_util = run()
num_epoch = 10
if __name__ == '__main__':
    cnet.train()
    cost = []
    accuracy = []
    loss_book = xlsxwriter.Workbook(filename='/path/to/fine_tuning_loss.xlsx')
    acc_book = xlsxwriter.Workbook(filename='/path/to/fine_tuning_result.xlsx')
    for epoch in range(num_epoch):
        run_util.adjust_learning_rate(optimizer=optimizer,
                                      epoch=epoch,
                                      epochs=num_epoch,
                                      lr=lr)
        run_util.train(model=cnet,
                       epoch=[epoch, num_epoch],
                       train_loader=train_loader,
                       device=device,
                       optimizer=optimizer,
                       criterion=criterion,
                       cost=cost)
        torch.save(cnet.state_dict(), '/path/to/fine_tuning.pkl'.format(epoch))
        loss_sheet = loss_book.add_worksheet('epoch_{}'.format(epoch))
        for i, loss in enumerate(cost):
            loss_sheet.write(0, i, loss)
        run_util.eval(model=cnet,
                      epoch=[epoch, num_epoch],
                      eval_loader=eval_loader,
                      device=device,
                      accuracy=accuracy)
        acc_sheet = acc_book.add_worksheet('epoch_{}'.format(epoch))
        for i, acc in enumerate(accuracy):
            acc_sheet.write(0, i, acc)
    loss_book.close()
    acc_book.close()
