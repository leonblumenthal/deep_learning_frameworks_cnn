import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, optimizers


### Hyperparameters

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.8
MOMENTUM = 0.9


### MNIST data loading

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Convert grayscale values from [0,255] to [0,1] and add depth dimension.
train_images = train_images[..., tf.newaxis].astype("float32") / 255
test_images = test_images[..., tf.newaxis].astype("float32") / 255
# Standardize images.
train_images = tf.image.per_image_standardization(train_images)
test_images = tf.image.per_image_standardization(test_images)


### Create network and set hyperparameters.

# TODO: GPU usage

# Simplest implementation for the desired network structure
net = models.Sequential(
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

# Stochastic gradient descent with momentum and learning rate decay
scheduler = optimizers.schedules.ExponentialDecay(
    LEARNING_RATE,
    decay_steps=6000,
    decay_rate=LEARNING_RATE_DECAY,
    staircase=True,
)
optimizer = optimizers.SGD(learning_rate=scheduler, momentum=MOMENTUM)

net.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer=optimizer,
    metrics='accuracy',
)

### Actual training and testing

net.fit(
    x=train_images,
    y=train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(test_images, test_labels),
)


### Network parameters saving and loading

net.save_weights('tf_net')
net.load_weights('tf_net')