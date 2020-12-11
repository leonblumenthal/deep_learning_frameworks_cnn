# Deep Learning Frameworks - Case Study: Convolutional Neural Network  

*TensorFlow vs. PyTorch*

## Dataset

The MNIST database of handwritten digits http://yann.lecun.com/exdb/mnist/
  
- grayscale [0, 255] images of single handwritten (28 x 28)
- labels [0, 9]
- 60000 training
- 10000 test

Transform grayscale values from [0, 255] to [0, 1].  
Add depth dimension.  
Standardize images (i.e. mean 0, std 1).

## Network structure

**(28 x 28 x 1)**

- Conv 3 x 3 x 1 → 16
- ReLu
  
**(26 x 26 x 16)**

- Max Pool 2 x 2
  
**(13 x 13 x 16])**

- Conv 4 x 4 x 16 → 32
- ReLu
  
**(10 x 10 x 32)**

- Avg Pool 2 x 2
- Dropout 0.3
  
**(5 x 5 x 32)**

- Conv 5 x 5 x 32 → 64
- ReLu
- Flatten
  
**(64)**

- Dense
- tanh
- Dropout 0.2
  
**(32)**

- Dense
- Softmax
  
**(10)**

## Training

**Epochs:** 10

**Batch size:** 32

**Loss function:** Cross entropy loss

**Optimizer:** SGD

**Learning rate:** 0.01

**Learning rate decay:** 0.8

**Momentum:** 0.9