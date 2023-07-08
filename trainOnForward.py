import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model import NetForward
from database import kWaveDatasetTest, kWaveDatasetTrain
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter('./log/runs{}'.format(len(os.listdir('./log'))))

device = torch.device("cuda:0")

# 导入数据


trainSet = kWaveDatasetTrain()
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=128, shuffle=True)

testSet = kWaveDatasetTest()
tsetLoader = torch.utils.data.DataLoader(testSet, batch_size=128, shuffle=True)

# 定义网络
net = NetForward().to(device)

# writer.add_graph(net, torch.ones(1, 88, 88).to(device))
# 定义损失及优化函数

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

n = 5
best_valid_loss = float('inf')
best_model_params = None
fla = 0
# 训练
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):

        labels, inputs = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.float(), labels.float()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # 更新最佳模型参数
        if loss < best_valid_loss:
            best_valid_loss = loss
            best_model_params = net.state_dict().copy()

        # 提前停止条件：连续 n 个 epoch 验证集上的损失没有改善
        if epoch > n and loss > best_valid_loss:
            break

        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        fla += 1

        writer.add_scalar('Loss/train_loss', running_loss, global_step=fla)

        print('fla %d loss: %.8f' %
              (fla, running_loss))

    # break

net.load_state_dict(best_model_params)
torch.save(net.state_dict(), 'NetForwardWeights.pth')
writer.close()
