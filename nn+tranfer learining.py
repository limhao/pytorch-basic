import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 大概是给导入relu使用的
import torchvision.datasets as datasets
import torch.optim as optim
device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# 修改模型结构
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 10),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
model.to(device)

# 定义超参数
learning_rate = 0.001
num_epoch = 10
batch_size = 64
# 定义优化器 和 损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
crition = nn.CrossEntropyLoss()


