import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 大概是给导入relu使用的
import torchvision.datasets as datasets
import torch.optim as optim

# 规定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义网络
class NN(nn.Module):
    def __init__(self, input_size, num_classses):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classses)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化网络
model = NN(input_size=784, num_classses=10).to(device=device)
# 超参数
batch_size = 64
learining_rate = 0.001
num_epoch = 1

# 定义 损失函数 和 优化器
optimizer = optim.Adam(model.parameters(), lr=learining_rate)
critierea = nn.CrossEntropyLoss()
# 加载数据 定义data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
# 训练模型 观察正确情况

for epoch in range(num_epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data = data.reshape(data.shape[0], -1)
        # 前向
        score = model(data)
        loss = critierea(score, target)
        losses.append(loss.item())
        # 后向
        optimizer.zero_grad()
        loss.backward()
        # step
        optimizer.step()
    mean_loss = sum(losses) / len(losses)
    print(f'loss at epoch {epoch} was {mean_loss:.5f}')




def check_accuracy(model, loader):
    total_acc = 0
    total_num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)
            score = model(x)
            _, pred = score.max(1)
            total_acc += (pred == y).sum()
            total_num += pred.size(0)

        print(f'accuary:{total_acc}/{total_num}')

check_accuracy(model, train_loader)
check_accuracy(model, test_loader)
