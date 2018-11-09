


import torchvision as tv
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
show = ToPILImage() # 可以把Tensor转成Image，方便可视化

transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])
# 训练集
trainset = tv.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
    './data',
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义函数来显示图像


import matplotlib.pyplot as plt
import numpy as np

# 定义函数来显示图像


# def imshow(img):
#     img = img / 2 + 0.5     # 非标准化
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 得到一些随机的训练图像
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# 显示图像
# imshow(tv.utils.make_grid(images))
# 输出类别
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid((images+1)/2)).resize((400,100))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().cuda()
print(net)

criterion  = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# torch.set_num_threads(8)
print("==")
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # 输入数据
        # torch.
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # net.to(device)
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # inputs.to(device)
        # labels.to(device)
        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
correct = 0
total = 0
for data in testloader:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torch.optim as optim
#
# class Net(nn.Module):
#     def __init__(self):
#         # nn.Module子类的函数必须在构造函数中执行父类的构造函数
#         # 下式等价于nn.Module.__init__(self)
#         super(Net, self).__init__()
#
#         # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         # 卷积层
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # 仿射层/全连接层，y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # 卷积 -> 激活 -> 池化
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         # reshape，‘-1’表示自适应
#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# net = Net()
# print(net)
# target = Variable(torch.arange(0,10).view(1,10))
# criterion = nn.MSELoss()
# #新建一个优化器，指定要调整的参数和学习率
# optimizer = optim.SGD(net.parameters(), lr = 0.01)
#
# optimizer.zero_grad()
# input = Variable(torch.randn(1, 1, 32, 32))
# # 计算损失
# output = net(input)
# loss = criterion(output, target.float())
# print(loss)
# loss.backward()
# optimizer.step()