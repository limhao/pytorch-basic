# -*- coding:utf-8 -*-
"""
@auther:warma
@software: PyCharm
@file: cnn.py
@time: 2021/12/14 5:06 下午
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 大概是给导入relu使用的
import torchvision.datasets as datasets
import torch.optim as optim

# 最扯淡的是 我用cuda跑出来的结果 比 苹果上面用m1的效果要好很多 好很多

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1*28*28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=2, padding=2, stride=2)
        # conv1 18*16*16
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # pooling 18*8*8
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=36, kernel_size=2, padding=1, stride=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # conv2 36*5*5 pooling 36*2*2
        self.fc1 = nn.Linear(in_features=144, out_features=10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pooling1(x)
        x = F.relu(self.conv2(x))
        x = self.pooling2(x)
        # 这里需要将x给展平
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
# 创建一个cnn网络
# class CNN(nn.Module):
#     def __init__(self, in_channels=1, num_classes=10):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc1(x)
#
#         return x



# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
num_epoch = 5
learning_rate = 0.01
batch_size = 64
# 初始化网络
model = CNN().to(device)

# 定义优化器 和 损失函数 要在初始化网络的后面
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 加载数据 定义data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    losses = []
    for epoch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # 得到预测值 forward loss 前面是pred 后面是target
        pred = model(data)
        loss = criterion(pred, target)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # 走一步
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f'epoch:{epoch} at loss {mean_loss}')


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for test_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            # predicition 64 * 1
            __, predicition = scores.max(1)
            num_correct += (predicition == y).sum()
            num_samples += predicition.size(0)
            #print(f'epoch {test_idx},with accuary at {float(epoch_correct) / float(epoch_samples) * 100:.2f}')


        print(f'got {num_correct}/{num_samples} with accuary {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
