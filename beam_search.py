# _*_ coding:utf-8 _*_
"""
@Software: flevoland15_code
@FileName: beam_search.py
@Date: 2022/12/14 21:42
@Author: caijianfeng
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import features_Dataset
from net_run import run

from network import Beam_search_Net

num_class = 0
features_num = 0
oned_net = Beam_search_Net(num_class=num_class,
                           features_num=features_num)

# In[2] data load(train and eval)
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, ), (0.229, ))
])
dataset_path = '/path/to/'
train_dataset = features_Dataset(dataset_path=dataset_path,
                                 transform=transform)

if torch.cuda.is_available():
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=4)
else:
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)

# In[3] generate network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
oned_net = oned_net.to(device)
print(oned_net)

# In[4] loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.Adam(params=oned_net.parameters(),
                       lr=lr,
                       weight_decay=1e-4)

# In[5] train
print('-----train-----')
run_util = run()
num_epoch = 10
if __name__ == '__main__':
    oned_net.train()
    cost = []
    accuracy = []

    for epoch in range(num_epoch):
        run_util.adjust_learning_rate(optimizer=optimizer,
                                      epoch=epoch,
                                      epochs=num_epoch,
                                      lr=lr)
        run_util.train(model=oned_net,
                       epoch=[epoch, num_epoch],
                       train_loader=train_loader,
                       device=device,
                       optimizer=optimizer,
                       criterion=criterion,
                       cost=cost)
        # Saved model parameter
        torch.save(oned_net.state_dict(), '/path/to/xxx.pkl'.format(epoch))


class Classifier:
    def __init__(self, net):
        self.net = net

    def __call__(self, data, labels):
        predict = self.net(data)
        _, predict = torch.max(predict.data, dim=1)
        acc = (predict == labels) / labels.size(0)
        return acc


# In[6] beam search
classifier = Classifier(net=oned_net)


def pad_zero_input(data, group):
    new_data = data.clone()
    for i in range(group):
        if group[i] == 1:
            new_data[:, :, i] = 0
    return new_data


def search_k(queue):
    min_k = None
    score = 1
    for group in queue:
        data = pad_zero_input(test_data, group)
        if classifier(data, labels) < score:
            min_k = group
            score = classifier(data, labels)
        return min_k if min_k else queue[0]


def search_max(queue):
    max_k = None
    score = 0
    for group in queue:
        data = pad_zero_input(test_data, group)
        if classifier(data, labels) > score:
            max_k = group
            score = classifier(data, labels)
        return max_k if max_k else queue[0]


theta = 0
k = 2
test_data = torch.randn((batch_size, 1, features_num))
labels = torch.randn((batch_size, num_class))
is_remove = [0 for _ in range(features_num)]
queue_k = [is_remove.copy()]
while features_num > theta:
    queue_k_new = queue_k.copy()
    for group in queue_k_new:
        for i in range(len(group)):
            if not group[i]:
                group_new = group.copy()
                group_new[i] = 1
                if len(queue_k) < k:
                    queue_k.append(group_new)
                else:
                    max_k = search_k(queue_k)
                    if classifier(pad_zero_input(test_data, max_k), labels) < classifier(
                            pad_zero_input(test_data, group_new), labels):
                        queue_k.remove(max_k)
                        queue_k.append(group_new)
    features_num -= 1

features_group = search_max(queue_k)
