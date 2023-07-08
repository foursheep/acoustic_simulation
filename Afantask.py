import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy.io import loadmat


class kWaveDataset(Dataset):
    def __init__(self):
        self.L0path = './L0'
        self.P0path = './P0'
        self.L0Dirlist = os.listdir(self.L0path)
        self.P0Dirlist = os.listdir(self.P0path)

    def __getitem__(self, index):
        L0 = loadmat(os.path.join(self.L0path, self.L0Dirlist[index]))['sensor_data']
        P0 = loadmat(os.path.join(self.P0path, self.P0Dirlist[index]))['p0']
        return L0, P0

    def __len__(self):
        return len(self.L0Dirlist)


class Net(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(13056, 7744)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = x.view(-1, 88, 88)
        return x


device = torch.device("cuda:0")

# 导入数据
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize()
])

trainset = kWaveDataset()

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = kWaveDataset()

tsetLoader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

# 定义网络
net = Net(64).to(device)

# 定义损失及优化函数

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005, )

b = torch.inf
a = 0
# 训练
for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.float(), labels.float()

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = 10e3 * criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('epoch %1d loss: %.6f' %
          (epoch + 1, running_loss / 20))

    if running_loss > b:
        a = a + 1
        if a > 10:
            break

    b = running_loss
# 测试
torch.save(net.state_dict(), 'net_weights.pth')
