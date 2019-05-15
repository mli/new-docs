# MXNet for PyTorch users in 10 minutes

[PyTorch](https://pytorch.org/) is a popular deep learning framework due to its easy-to-understand API and its completely imperative approach. Apache MXNet includes the Gluon API which gives you the simplicity and flexibility of PyTorch and allows you to hybridize your network to leverage performance optimizations of the symbolic graph.

In the next 10 minutes, we'll do a quick comparison between the two frameworks and show how small the learning curve can be when switching from PyTorch to Apache MXNet.

## Installation

PyTorch uses conda for installation by default, for example:

```{.python .input}
# !conda install pytorch-cpu -c pytorch
```

For MXNet we use pip:

```{.python .input}
# !pip install mxnet
```

To install Apache MXNet with GPU support, you need to specify CUDA version. For example, the snippet below will install Apache MXNet with CUDA 9.2 support:

```{.python .input}
# !pip install mxnet-cuda92
```

## Data manipulation

Both PyTorch and Apache MXNet relies on multidimensional matrices as a data sources. While PyTorch follows Torch's naming convention and refers to multidimensional matrices as "tensors", Apache MXNet follows NumPy's conventions and refers to them as "NDArrays".

In the code snippets below, we create a two-dimensional matrix where each element is initialized to 1. We show how to add 1 to each element of matrices and print the results.

**PyTorch:**

```{.python .input}
import torch

x = torch.ones(5,3)
y = x + 1
y
```

**MXNet:**

```{.python .input}
from mxnet import nd

x = nd.ones((5,3))
y = x + 1
y
```

The main difference apart from the package name is that the MXNet's shape input parameter needs to be passed as a tuple enclosed in parentheses as in NumPy.

Both frameworks support multiple functions to create and manipulate tensors / NDArrays. You can find more of them in the documentation.

## Model training

After covering the basics of data creation and manipulation, let's dive deep and compare how model training is done in both frameworks. In order to do so, we are going to solve image classification task on MNIST data set using Multilayer Perceptron (MLP) in both frameworks. We divide the task in 4 steps.

### 1 --- Read data

The first step is to obtain the data. We download the MNIST data set from the web and load it into memory so that we can read batches one by one.

**PyTorch:**

```{.python .input}
from torchvision import datasets, transforms

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.13,), (0.31,))])
pt_train_data = torch.utils.data.DataLoader(datasets.MNIST(
    root='.', train=True, download=True, transform=trans),
    batch_size=128, shuffle=True, num_workers=4)
```

**MXNet:**

```{.python .input}
from mxnet import gluon
from mxnet.gluon.data.vision import datasets, transforms

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(0.13, 0.31)])
mx_train_data = gluon.data.DataLoader(
    datasets.MNIST(train=True).transform_first(trans),
    batch_size=128, shuffle=True, num_workers=4)
```

Both frameworks allows you to download MNIST data set from their sources and specify that only training part of the data set is required.

The main difference between the code snippets is that MXNet uses [transform_first](http://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.data.Dataset.html) method to indicate that the data transformation is done on the first element of the data batch, the MNIST picture, rather than the second element, the label.

### 2 --- Creating the model

Below we define a Multilayer Perceptron (MLP) with a single hidden layer
and 10 units in the output layer.

**PyTorch:**

```{.python .input}
import torch.nn as pt_nn

pt_net = pt_nn.Sequential(
    pt_nn.Linear(28*28, 256),
    pt_nn.ReLU(),
    pt_nn.Linear(256, 10))
```

**MXNet:**

```{.python .input}
import mxnet.gluon.nn as mx_nn

mx_net = mx_nn.Sequential()
mx_net.add(mx_nn.Dense(256, activation='relu'),
           mx_nn.Dense(10))
mx_net.initialize()
```

We used the Sequential container to stack layers one after the other in order to construct the neural network. Apache MXNet differs from PyTorch in the following ways:

* In PyTorch you have to specify the input size as the first argument of the `Linear` object. Apache MXNet provides an extra flexibility to network structure by automatically inferring the input size after the first forward pass.

* In Apache MXNet you can specify activation functions directly in fully connected and convolutional layers.

* After the model structure is defined, Apache MXNet requires you to explicitly call the model initialization function.

With a Sequential block, layers are executed one after the other. To have a different execution model, with PyTorch you can inherit from `nn.Module` and then customize how the `.forward()` function is executed. Similarly, in Apache MXNet you can inherit from [nn.Block](http://beta.mxnet.io/api/gluon/mxnet.gluon.nn.Block.html) to achieve similar results.

### 3 --- Loss function and optimization algorithm

The next step is to define the loss function and pick an optimization algorithm. Both PyTorch and Apache MXNet provide multiple options to chose from, and for our particular case we are going to use the cross-entropy loss function and the Stochastic Gradient Descent (SGD) optimization algorithm.

**PyTorch:**

```{.python .input}
pt_loss_fn = pt_nn.CrossEntropyLoss()
pt_trainer = torch.optim.SGD(pt_net.parameters(), lr=0.1)
```

**MXNet:**

```{.python .input}
mx_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mx_trainer = gluon.Trainer(mx_net.collect_params(),
                           'sgd', {'learning_rate': 0.1})
```

The code difference between frameworks is small. The main difference is that in Apache MXNet we use [Trainer](http://beta.mxnet.io/api/gluon/mxnet.gluon.Trainer.html) class, which accepts optimization algorithm as an argument. We also use [.collect_params()](https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.Block.collect_params.html) method to get parameters of the network.

### 4 --- Training

Finally, we implement the training algorithm. Note that the results for each run
may vary because the weights will get different initialization values and the
data will be read in a different order due to shuffling.

**PyTorch:**

```{.python .input}
import time

for epoch in range(5):
    total_loss = .0
    tic = time.time()
    for X, y in pt_train_data:
        pt_trainer.zero_grad()
        loss = pt_loss_fn(pt_net(X.view(-1, 28*28)), y)
        loss.backward()
        pt_trainer.step()
        total_loss += loss.mean()
    print('epoch %d, avg loss %.4f, time %.2f' % (
        epoch, total_loss/len(pt_train_data), time.time()-tic))
```

**MXNet:**

```{.python .input}
from mxnet import autograd

for epoch in range(5):
    total_loss = .0
    tic = time.time()
    for X, y in mx_train_data:
        with autograd.record():
            loss = mx_loss_fn(mx_net(X), y)
        loss.backward()
        mx_trainer.step(batch_size=128)
        total_loss += loss.mean().asscalar()
    print('epoch %d, avg loss %.4f, time %.2f' % (
        epoch, total_loss/len(mx_train_data), time.time()-tic))
```

Some of the differences in Apache MXNet when compared to PyTorch are as follows:

* In Apache MXNet, you don't need to flatten the 4-D input into 2-D when feeding the data into forward pass.

* In Apache MXNet, you need to perform the calculation within the [autograd.record()](https://beta.mxnet.io/api/gluon-related/_autogen/mxnet.autograd.record.html) scope so that it can be automatically differentiated in the backward pass.

* It is not necessary to clear the gradient every time as with PyTorch's `trainer.zero_grad()` because by default the new gradient is written in, not accumulated.

* You need to specify the update step size (usually batch size) when performing [step()](https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.Trainer.step.html) on the trainer.

* You need to call [`.asscalar()`](https://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.NDArray.asscalar.html) to turn a multidimensional array into a scalar.

* In this sample, Apache MXNet is twice as fast as PyTorch. Though you need to be cautious with such toy comparisons.

## Conclusion

As we saw above, Apache MXNet Gluon API and PyTorch are similar in use.

## Recommended Next Steps

While Apache MXNet Gluon API is very similar to PyTorch, there are some extra functionality using which can make your code even faster. 

* Check out [Hybridize tutorial](https://beta.mxnet.io/guide/packages/gluon/hybridize.html) to learn how to write imperative code which can be converted to symbolic one. 

* Also, check out how to extend Apache MXNet with your own [custom layers](https://beta.mxnet.io/guide/extend/custom_layer.html).