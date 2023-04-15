"""
created on:2022/11/13 15:13
@author:caijianfeng
"""
import os.path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_preprocess import Data
from network import HetNet
from dataset import PolSAR_Dataset
from net_run import run
import sys
import xlsxwriter

sys.path.append(os.getcwd())  # Add the current working directory to your system path

# In[1] Generate data: Get the set of features for each data point and the T-matrix of the data blocks
data_path = ['/path/to/T_R.xlsx',
             '/path/to/T_I.xlsx']
target_path = ['/path/to/T/',
               '/path/to/F/']
contrastive_list_path = '/path/to/contrastive.txt'
patch_size = [15, 15]
if not os.path.exists(contrastive_list_path):
    data_util = Data()
    data_util.save_contrastive_data_origin(data_path=data_path,
                                           target_path=target_path,
                                           contrastive_list_path=contrastive_list_path,
                                           patch_size=patch_size)
# In[2] 构造数据集
print('-----CL train begin(lr adjust)-----')
dataset_path = '/path/to'
batch_size = 4096
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, ), (0.229, ))
])
contrastive_train_dataset = PolSAR_Dataset(dataset_path=dataset_path,
                                           mode='contrastive',
                                           transform=transform)
if torch.cuda.is_available():
    contrastive_train_dataloader = DataLoader(dataset=contrastive_train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8)
else:
    contrastive_train_dataloader = DataLoader(dataset=contrastive_train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

# In[3] Construct the network: Construct the heterogeneous network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = HetNet(T=0.07, device=device)
# net = HetNet(T=0.07)
net = net.to(device=device)

criterion = torch.nn.CrossEntropyLoss()
lr = 0.01
# optimizer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.5)
optimizer = optim.SGD(params=net.parameters(),
                      lr=lr,
                      momentum=0.9,
                      weight_decay=1e-4)

# In[4] CL training
run_util = run()
num_epoch = 20
if __name__ == '__main__':
    net.train()
    cost = []
    loss_book = xlsxwriter.Workbook(filename='/path/to/CL_loss.xlsx')
    for epoch in range(num_epoch):
        run_util.adjust_learning_rate(optimizer=optimizer,
                                      epoch=epoch,
                                      epochs=num_epoch,
                                      lr=lr)
        run_util.contrastive(model=net,
                             epoch=[epoch, num_epoch],
                             train_loader=contrastive_train_dataloader,
                             device=device,
                             optimizer=optimizer,
                             criterion=criterion,
                             cost=cost)
        torch.save(net.state_dict(), '/path/to/CL_{}.pkl'.format(epoch))
    loss_sheet = loss_book.add_worksheet('epoch')
    for i, loss in enumerate(cost):
        loss_sheet.write(0, i, loss)
    loss_book.close()
