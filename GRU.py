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






# 超参数
input_size =28
sequence_length = 28
hidden_size = 256
num_layers = 2
num_classses = 10
batch_size = 64
learining_rate = 0.001
num_epoch = 2

# 定义rnn 28个时间序列 每个序列有28个特征 MNIST
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classses):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classses)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# 初始化网络
model = RNN(input_size, hidden_size, num_layers, num_classses).to(device=device)

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
        data = data.to(device).squeeze(1)
        target = target.to(device)

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
            x = x.to(device).squeeze(1)
            y = y.to(device)

            score = model(x)
            _, pred = score.max(1)
            total_acc += (pred == y).sum()
            total_num += pred.size(0)

        print(f'accuary:{total_acc}/{total_num}with accuracy {float(total_acc) / float(total_num) * 100:.2f}')

check_accuracy(model, train_loader)
check_accuracy(model, test_loader)