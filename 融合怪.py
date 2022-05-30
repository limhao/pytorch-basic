# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# 创建一个全连接网络
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 28*28 = 784
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建一个cnn网络
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("saveing checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("load checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
# model = NN(784, 10)
#
# x = torch.randn(64, 784)
# print(model(x).shape)
# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 超参数
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epoch = 10
load_model = False

import sys

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
# load pertrain model & modify it
model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# 修改模型结构
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 10),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
model.to(device)
# print(model)
# 加载数据
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 初始化网络
# model = CNN().to(device)

# 损失 和 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))
# 训练

for epoch in range(num_epoch):
    losses = []
    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, target) in enumerate(train_loader):
        # 从cuda里面那数据 可以的话
        data = data.to(device=device)
        target = target.to(device=device)

        # forward
        score = model(data)
        loss = criterion(score, target)
        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient desent or adam step
        optimizer.step()

    mean_loss = sum(losses)/len(losses)
    print(f'loss at epoch {epoch} was {mean_loss:.5f}')

# 检查正确率 测试和训练

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("check accuracy on training data")
    else:
        print("check accuracy on test data")
    num_correct = 0
    num_samples = 0
    # 设置为验证模式
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            y = y.to(device)
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # print(scores)
            # y 大小为 64*1 的一维数组
            # 前面是索引 不关心 max（1）返回每一行最大值组成的一维数组
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            # axis = 0，返回该二维矩阵的行数
            # print(predictions,y)
            num_samples += predictions.size(0)

        print(f'{num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
