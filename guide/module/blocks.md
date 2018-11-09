
The code examples in this section assume both ``nd`` and ``nn`` modules are imported.


from mxnet import nd
from mxnet.gluon import nn

doctest::

>>> from mxnet import nd
>>> from mxnet.gluon import nn

The :py:class:`Block` class provided in the :py:mod:`mxnet.gluon.nn` package
(also available in :py:mod:`mxnet.gluon`) is the base class any neural network
layers and models. It can be used to implement either a neural network layer or
to consturct a multi-layer neural network model. For example, implement a
customized dense layer:


>>> class Dense(nn.Block):
...     # units: number of output units; in_units: number of input units
...     def __init__(self, units, in_units, **kwargs):
...         super(Dense, self).__init__(**kwargs)
...         self.weight = self.params.get('weight', shape=(in_units, units))
...         self.bias = self.params.get('bias', shape=(units,))
...     def forward(self, x):
...         linear = nd.dot(x, self.weight.data()) + self.bias.data()
...         return nd.relu(linear)

Implement a two-layer perceptron by using the build-in dense layer
:py:class:`Dense`:

doctest::

>>> class MLP(nn.Block):
...     def __init__(self, **kwargs):
...         super(MLP, self).__init__(**kwargs)
...         self.hidden = nn.Dense(10)
...         self.output = nn.Dense(2)
...     def forward(self, x):
...         y = nd.relu(self.hidden(x))
...         return self.output(y).sum()
>>> net = MLP()
>>> print(net)
MLP(
  (hidden): Dense(None -> 10, linear)
  (output): Dense(None -> 2, linear)
)
