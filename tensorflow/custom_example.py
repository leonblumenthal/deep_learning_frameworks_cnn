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

# Shuffle images and split into mini batches.
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(len(train_images))
    .batch(BATCH_SIZE)
)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
    BATCH_SIZE
)


### Create network and set hyperparameters.

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
    decay_steps=len(train_ds),
    decay_rate=LEARNING_RATE_DECAY,
    staircase=True,
)
optimizer = optimizers.SGD(learning_rate=scheduler, momentum=MOMENTUM)

loss_function = losses.SparseCategoricalCrossentropy()

# Setup metrics for training and testing.
train_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


### Train and test helper functions

def train():
    total = 0
    # Iterate through all mini batches.
    for images, labels in train_ds:
        # Record computations on tape.
        with tf.GradientTape() as tape:
            # Compute loss based on outputs and labels in training mode.
            loss = loss_function(labels, net(images, training=True))
        # Compute gradients and update parameters.
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))

        # Print progress.
        train_loss(loss)
        total += len(labels)
        s = f'Training {total}/{len(train_labels)} - loss: {train_loss.result():.4f}'
        print(s, end='\r')

    print(s)
    train_loss.reset_states()


def test():
    total = 0

    # Iterate through all mini batches ignoring gradients.
    for images, labels in test_ds:
        # Compute outputs in test mode.
        outputs = net(images, training=False)

        # Print progress.
        total += len(images)
        test_accuracy(labels, outputs)
        s = f'Test {total}/{len(test_labels)} - accuracy: {test_accuracy.result():.4f}'
        print(s, end='\r')

    print(s)
    test_accuracy.reset_states()


### Actual training and testing

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')

    train()
    test()


### Network parameters saving and loading

net.save_weights('tf_net')
net.load_weights('tf_net')