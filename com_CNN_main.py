# _*_ coding:utf-8 _*_
"""
@Software: code
@FileName: com_CNN_main.py
@Date: 2023/2/18 20:37
@Author: caijianfeng
"""
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from network import DClass_Net
from dataset import PolSAR_Dataset
from torch.utils.data import DataLoader
from net_run import run
import xlsxwriter
from data_preprocess import Data
import os

# In[1] generate train dataset
contrastive_path = '/path/to/contrastive.txt'
train_list_path = '/path/to/train.txt'
eval_list_path = '/path/to/eval.txt'
predict_list_path = '/path/to/predict.txt'
patch_size = [15, 15]  # in this article is [15, 15]; you can change it
data_path = ['/path/to/TR.xlsx', '/path/to/TI.xlsx']
label_path = '/path/to/label.xlsx'
target_path = '/path/to/'
num_class = 0
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

# In[2] data load(train and eval)
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, ), (0.229, ))
])
dataset_path = '../path/to/'
train_dataset = PolSAR_Dataset(dataset_path=dataset_path,
                               mode='train',
                               transform=transform)
eval_dataset = PolSAR_Dataset(dataset_path=dataset_path,
                              mode='eval',
                              transform=transform)
if torch.cuda.is_available():
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=4)
    eval_loader = DataLoader(eval_dataset,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=4)
else:
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)
    eval_loader = DataLoader(eval_dataset,
                             shuffle=False,
                             batch_size=batch_size)

# In[3] generate network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnet = DClass_Net(num_class=num_class)
cnet = cnet.to(device)
print(cnet)

# In[4] loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.Adam(params=cnet.parameters(),
                       lr=lr,
                       weight_decay=1e-4)

# In[5] train
print('-----train-----')
run_util = run()
num_epoch = 10
if __name__ == '__main__':
    cnet.train()
    cost = []
    accuracy = []

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
        # Saved model parameter
        torch.save(cnet.state_dict(), '/path/to/xxx.pkl'.format(epoch))
        loss_book = xlsxwriter.Workbook(
            filename='/path/to/CNN_loss.xlsx')
        loss_sheet = loss_book.add_worksheet('epoch_{}'.format(epoch))
        for i, loss in enumerate(cost):
            loss_sheet.write(0, i, loss)
        loss_book.close()
        run_util.eval(model=cnet,
                      epoch=[epoch, num_epoch],
                      eval_loader=eval_loader,
                      device=device,
                      accuracy=accuracy)
        acc_book = xlsxwriter.Workbook(filename='/path/to/CNN_result.xlsx')
        acc_sheet = acc_book.add_worksheet('epoch_{}'.format(epoch))
        for i, acc in enumerate(accuracy):
            acc_sheet.write(0, i, acc)
        acc_book.close()
