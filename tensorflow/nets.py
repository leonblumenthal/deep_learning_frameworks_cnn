import tensorflow as tf
from tensorflow.keras import layers, models


# Network with explicit activation functions and only layers
class Net1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(16, 3)
        self.relu1 = layers.ReLU()
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(32, 4)
        self.relu2 = layers.ReLU()
        self.pool2 = layers.AvgPool2D(2)
        self.dropout1 = layers.Dropout(0.3)
        self.conv3 = layers.Conv2D(64, 5)
        self.relu3 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(32)
        self.tanh = layers.Activation('tanh')  # tanh layer does not exist.
        self.dropout2 = layers.Dropout(0.2)
        self.fc2 = layers.Dense(10)
        self.softmax = layers.Softmax(1)

    def call(self, x):
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


# Network with explicit activation functions and only stateful layers
class Net2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(16, 3)
        self.conv2 = layers.Conv2D(32, 4)
        self.dropout1 = layers.Dropout(0.3)
        self.conv3 = layers.Conv2D(64, 5)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(32)
        self.dropout2 = layers.Dropout(0.2)
        self.fc2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, 2, 1, 'VALID')
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = tf.nn.avg_pool2d(x, 2, 1, 'VALID')
        x = self.dropout1(x)
        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = tf.nn.tanh(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = tf.nn.softmax(x, 1)

        return x


# Network with inline activation functions and only stateful layers
class Net3(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(16, 3, activation='relu')
        self.conv2 = layers.Conv2D(32, 4, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.conv3 = layers.Conv2D(64, 5, activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(32, activation='tanh')
        self.dropout2 = layers.Dropout(0.2)
        self.fc2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.max_pool2d(x, 2, 1, 'VALID')
        x = self.conv2(x)
        x = tf.nn.avg_pool2d(x, 2, 1, 'VALID')
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = tf.nn.softmax(x, 1)

        return x


# Sequential with explicit activation functions
net4 = models.Sequential(
    [
        layers.Conv2D(16, 3),
        layers.ReLU(),
        layers.MaxPool2D(2),
        layers.Conv2D(32, 4),
        layers.ReLU(),
        layers.AvgPool2D(2),
        layers.Dropout(0.3),
        layers.Conv2D(64, 5),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(32),
        layers.Activation('tanh'),  # tanh layer does not exist.
        layers.Dropout(0.2),
        layers.Dense(10),
        layers.Softmax(),
    ]
)


# Sequential with inline activation functions
net5 = models.Sequential(
    [
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPool2D(2),
        layers.Conv2D(32, 4, activation='relu'),
        layers.AvgPool2D(2),
        layers.Dropout(0.3),
        layers.Conv2D(64, 5, activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(10),
        layers.Softmax(),
    ]
)