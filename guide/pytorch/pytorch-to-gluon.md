# MXNet for PyTorch users in 10 minutes

[PyTorch](https://pytorch.org/) has quickly established itself as one of the
most popular deep learning framework due to its easy-to-understand API and its
completely imperative approach. But you might not be aware that MXNet includes
the [Gluon API](https://gluon-crash-course.mxnet.io/) which gives you the
simplicity and flexibility of PyTorch, whilst allowing you to hybridize your
network to leverage performance optimizations of the symbolic graph.

In the next 10 minutes, we'll show you a quick comparison between the two
frameworks and show you how small the learning curve can be when switching from
one to the other. We use the example of image classification on MNIST dataset.

## Installation

PyTorch uses conda for installation by default, for example:

```bash
conda install pytorch-cpu -c pytorch
```

For MXNet we use pip:

```bash
pip install mxnet
```

## Multidimensional matrix

For multidimensional matrices, PyTorch follows Torch's naming convention and
refers to "tensors". MXNet follows NumPy's conventions and refers to
"ndarrays". Here we create a two-dimensional matrix where each element is
initialized to 1. Then we add 1 to each element and print.

PyTorch:

```python
import torch
x = torch.ones(5,3)
y = x + 1
print(y)
```

```
 2  2  2
 2  2  2
 2  2  2
 2  2  2
 2  2  2
[torch.FloatTensor of size 5x3]
```

MXNet:

```python
from mxnet import nd
x = nd.ones((5,3))
y = x + 1
print(y)
```

```
[[2. 2. 2.]
 [2. 2. 2.]
 [2. 2. 2.]
 [2. 2. 2.]
 [2. 2. 2.]]
<NDArray 5x3 @cpu(0)>
```

The main difference apart from the package name is that the MXNet's shape input
parameter needs to be passed as a tuple enclosed in parentheses as in NumPy.

## Model training

Let's look at a slightly more complicated example below. Here we create a
Multilayer Perceptron (MLP) to train a model on the MINST data set. We divide
the experiment into 4 sections.

## 1 - Read data

We download the MNIST data set and load it into memory so that we can read
batches one by one.

PyTorch:

```python
import torch
from torchvision import datasets, transforms

train_data = torch.utils.data.DataLoader(
    datasets.MNIST(train=True, transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.13,), (0.31,))])),
    batch_size=128, shuffle=True, num_workers=4)
```

MXNet:

```python
from mxnet import gluon
from mxnet.gluon.data.vision import datasets, transforms

train_data = gluon.data.DataLoader(
    datasets.MNIST(train=True).transform_first(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])),
	batch_size=128, shuffle=True, num_workers=4)
```

The main difference here is that MXNet uses transform_first to indicate that the data transformation is done on the first element of the data batch, the MNIST picture, rather than the second element, the label.

## 2 — Creating the model

Below we define a Multilayer Perceptron (MLP) with a single hidden layer and 10 units in the output layer.

PyTorch:

```python
from torch import nn

net = nn.Sequential(
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 10))
```

MXNet:

```python
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize()
```

We used the Sequential container to stack layers one after the other in order to construct the neural network. MXNet differs from PyTorch in the following ways:

* In MXNet, there is no need to specify the input size, it will be automatically inferred.

* You can specify activation functions directly in fully connected and convolutional layers.

* You need to create a name_scope to attach a unique name to each layer: this is needed to save and reload models later.

* You need to explicitly call the model initialization function.

With a Sequential block, layers are executed one after the other. To have a different execution model, with PyTorch you can inherit nn.Module and then customize how the .forward() function is executed. Similarly, in MXNet you can inherit nn.Block to achieve similar results.

## 3 — Loss function and optimization algorithm

PyTorch:

```python
loss_fn = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

MXNet:

```python
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
```

Here we pick a cross-entropy loss function and use the Stochastic Gradient
Descent algorithm with a fixed learning rate of 0.1.

## 4 — Training

Finally we implement the training algorithm. Note that the results for each run may vary because the weights will get different initialization values and the data will be read in a different order due to shuffling.

PyTorch:

```python
from time import time
for epoch in range(5):
    total_loss = .0
    tic = time()
    for X, y in train_data:
        X, y = torch.autograd.Variable(X), torch.autograd.Variable(y)
        trainer.zero_grad()
        loss = loss_fn(net(X.view(-1, 28*28)), y)
        loss.backward()
        trainer.step()
        total_loss += loss.mean()
    print('epoch *%*d, avg loss *%.4*f, time *%.2*f' % (
        epoch, total_loss/len(train_data), time()-tic))
```

```bash
epoch 0, avg loss 0.3251, time 3.71
epoch 1, avg loss 0.1509, time 4.05
epoch 2, avg loss 0.1057, time 4.07
epoch 3, avg loss 0.0820, time 3.70
epoch 4, avg loss 0.0666, time 3.63
```

MXNet:

```python
from time import time
for epoch in range(5):
    total_loss = .0
    tic = time()
    for X, y in train_data:
        with mx.autograd.record():
	        loss = loss_fn(net(X.flatten()), y)
        loss.backward()
        trainer.step(batch_size=128)
        total_loss += loss.mean().asscalar()
    print('epoch *%*d, avg loss *%.4*f, time *%.2*f' % (
        epoch, total_loss/len(train_data), time()-tic))
```

```bash
epoch 0, avg loss 0.3162, time 1.59
epoch 1, avg loss 0.1503, time 1.49
epoch 2, avg loss 0.1073, time 1.46
epoch 3, avg loss 0.0830, time 1.48
epoch 4, avg loss 0.0674, time 1.75
```

Some of the differences in MXNet when compared to PyTorch are as follows:

* You don't need to put the input into Variable *(This is not necessary anymore
  since PyTorch 0.4.0)*, but you need to perform the calculation within
  the `mx.autograd.record()` scope so that it can be automatically differentiated
  in the backward pass.

* It is not necessary to clear the gradient every time as with Pytorch's
  `trainer.zero_grad()` because by default the new gradient is written in, not
  accumulated.

* You need to specify the update step size (usually batch size) when
  performing `step()` on the trainer.

* You need to call `.asscalar()` to turn a multidimensional array into a scalar.

* In this sample, MXNet is twice as fast as PyTorch. Though you need to be
  cautious with such toy comparisons.
