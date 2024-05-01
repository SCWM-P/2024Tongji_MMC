import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 第一层卷积
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 第二层卷积
        self.fc1 = torch.nn.Linear(32 * 7 * 7, 128)  # 全连接层
        self.fc2 = torch.nn.Linear(128, 10)  # 输出层
        self.pool = torch.nn.MaxPool2d(2, 2)  # 池化层
        self.relu = torch.nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 应用第一层卷积+激活+池化
        x = self.pool(self.relu(self.conv2(x)))  # 应用第二层卷积+激活+池化
        x = torch.flatten(x, 1)  # 扁平化处理
        x = self.relu(self.fc1(x))  # 应用第一个全连接层
        x = self.fc2(x)  # 应用输出层
        return x