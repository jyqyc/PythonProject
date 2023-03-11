# 使用pytorch完成手写数字的识别
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np
BATCH_SIZE = 128
TEST_BATCH_SIZE=1000


# 1.准备数据集

def get_dataloader(train=True,batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))  # mean和std的形状和通道数相同
    ])
    dataset = MNIST(root='./data', train=True, transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# 2.构架模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        # 1.修改形状
        x = input.view([-1, 1 * 28 * 28])
        # 2.进行全连接的操作
        x = self.fc1(x)
        # 3.进行激活函数的处理
        x = F.relu(x)
        # 4.输出层
        out = self.fc2(x)

        return F.log_softmax(out, dim=-1)


model = MnistModel()
optimizer = Adam(model.parameters(), lr=0.001)
if os.path.exists('./venv/model.pkl'):
    model.load_state_dict(torch.load('./venv/model.pkl'))
    optimizer.load_state_dict(torch.load('./venv/optimizer.pkl'))


def train(epoch):
    data_loader = get_dataloader()
    for idx, (input, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)  # 调用模型，得到n预测值
        loss = F.nll_loss(output, target)
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
        if idx % 10 == 0:
            print(epoch, idx, loss.item())

        # 模型的保存
        if idx % 100 == 0:
            torch.save(model.state_dict(), './venv/model.pkl')
            torch.save(optimizer.state_dict(), './venv/optimizer.pkl')

def test():
    loss_list=[]
    acc_list=[]
    test_dataloader=get_dataloader(train=False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output=model(input)
            cur_loss=F.nll_loss(output,target)
            loss_list.append(cur_loss)
            pred=output.max(dim=-1)[-1]
            cur_acc=pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print(np.mean(acc_list),np.mean(loss_list))


if __name__ == "__main__":
    test()
