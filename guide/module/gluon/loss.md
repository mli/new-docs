# Loss functions

A loss function measure the difference between predicted outputs and the
according label. We can then run back propogation to compute the gradients.

Here is a basic usage of a loss function.

.. doctest::

```python
from mxnet import nd, autograd
from mxnet.gluon import nn, loss as gloss  # avoid using common name loss
loss = gloss.L2Loss()  # get an instance of the L2 Loss function
x = nd.random.uniform(shape=(4, 2))  # input example
y = nd.random.uniform(shape=(4,))  # label
net = nn.Sequential()  # construct a single-layer perceptron
net.add(nn.Dense(1))
net.initialize()
with autograd.record():
    # compute the loss between the predicted results and the label
    l =  loss(net(x), y)
    print(l.shape)
l.backward()  # run back propogation
print(net[0].weight.grad().shape)  # print the shape of the weight gradient
```
