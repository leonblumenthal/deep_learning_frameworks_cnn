import torch
import torch.nn as nn
import torch.nn.functional as F


# Network with only layers
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 4)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


# Network with only stateful layers
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 4)
        self.dropout1 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.softmax(x, 1)

        return x


# Sequential
net3 = nn.Sequential(
    nn.Conv2d(1, 16, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 4),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Dropout(0.3),
    nn.Conv2d(32, 64, 5),
    nn.ReLU(),
    nn.Flatten(),
    nn.Tanh(),
    nn.Linear(64, 32),
    nn.Dropout(0.2),
    nn.Linear(32, 10),
    nn.Softmax(1),
)