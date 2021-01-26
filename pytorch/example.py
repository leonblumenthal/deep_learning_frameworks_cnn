import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets


### Hyperparameters

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.8
MOMENTUM = 0.9


### MNIST data loading

# Load data and standardize images.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

# Shuffle images and split into mini batches.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


### Create network and set hyperparameters.

# Simplest implementation for the desired network structure
net = nn.Sequential(
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
    nn.Linear(64, 32),
    nn.Tanh(),
    nn.Dropout(0.2),
    nn.Linear(32, 10),
    nn.Softmax(1),
)

# Choose GPU if available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Stochastic gradient descent with momentum and learning rate decay
optimizer = optim.SGD(net.parameters(), LEARNING_RATE, MOMENTUM)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LEARNING_RATE_DECAY)

loss_function = nn.CrossEntropyLoss()


### Train and test helper functions

def train():
    # Set network to training mode.
    net.train()

    total = total_loss = 0

    # Iterate through all mini batches.
    for images, labels in train_loader:
        # Move tensors to GPU if available.
        images, labels = images.to(device), labels.to(device)

        # Compute loss based on outputs and labels.
        loss = loss_function(net(images), labels)
        # Compute gradients and update parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress.
        total += len(labels)
        total_loss += loss.item()
        s = f'Training {total}/{len(train_loader.dataset)} - loss: {total_loss / total:.4f}'
        print(s, end='\r')

    scheduler.step()

    print(s)


def test():
    # Set network to test mode.
    net.eval()

    total = correct = 0

    # Iterate through all mini batches ignoring gradients.
    with torch.no_grad():
        for images, labels in test_loader:
            # Move tensors to GPU if available.
            images = images.to(device)
            labels = labels.to(device)

            # Compute predictions based on outputs and labels.
            predictions = net(images).argmax(dim=1)

            # Print progress.
            total += len(images)
            correct += (predictions == labels).sum().item()
            s = f'Test {total}/{len(test_loader.dataset)} - accuracy: {correct / total:.4f}'
            print(s, end='\r')

    print(s)


### Actual training and testing

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')

    train()
    test()


### Network parameters saving and loading

torch.save(net.state_dict(), 'pt_net')
net.load_state_dict(torch.load('pt_net'))
