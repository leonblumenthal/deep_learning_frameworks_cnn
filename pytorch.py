import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets


### MNIST data loading

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)


### Network structure.

class Net(nn.Module):
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
        x = F.softmax(x, dim=1)

        return x


### Create network and set hyperparameters.

# Choose GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net.to(device)

optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=1, gamma=0.8)


### Train and test helper functions

def train():
    # Set network to training mode.
    net.train()

    total_loss = 0
    # Iterate through all mini batches.
    for inputs, labels in train_loader:
        # Move data to device.
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Compute outputs and loss based on labels.
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item()
        # Compute gradients and update parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Training loss: {total_loss}')


def test():
    # Set network to test mode.
    net.eval()

    corrects = 0
    # Iterate through all mini batches ignoring gradients.
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to device.
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Compute outputs and correct predictions based on labels.
            outputs = net(inputs)
            predictions = outputs.argmax(dim=1)
            corrects += (predictions == labels).sum().item()

    total = len(test_loader.dataset)
    print(f'Test corrects {corrects} / {total} ({corrects / total})')


### Actual training loop

for epoch in range(1, 21):
    print(f'# Epoch {epoch}')

    train()
    test()
    
    # Adjust learning rate.
    scheduler.step()


### Model saving and loading

torch.save(net.state_dict(), 'net.pt')
# net.load_state_dict(torch.load('net.pt'))
