import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 大概是给导入relu使用的
import torchvision.datasets as datasets
import torch.optim as optim

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model
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
model = NN(784, 10).to(device)

# 定义 optim 和 loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criter = nn.CrossEntropyLoss()

# 超参数
num_epoch = 10
learning_rate = 0.001
batch_size = 64
load_model = True

# load model save checkpoint 加载和保存模型
# 下面是保存模型权重 和 优化器权重
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("saveing checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("load checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))


# 加载数据
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    losses = []
    # 通过state_dict 保存目前的情况
    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, target) in enumerate(train_loader):
        # 从cuda里面那数据 可以的话
        data = data.to(device=device)
        target = target.to(device=device)
        data = data.reshape(data.shape[0], -1)
        # forward
        score = model(data)
        loss = criter(score, target)
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
            x = x.reshape(x.shape[0], -1)

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