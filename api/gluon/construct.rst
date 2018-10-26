Containers
==========

Containers are used to construct multi-layer neural network. For example, use
``nn.Sequential`` to sequentially combine two dense layers:

.. doctest::

    >>> from mxnet.gluon import nn
    >>> net = nn.Sequential()
    >>> net.add(nn.Dense(10, activation='relu'), nn.Dense(4))
    >>> print(net)
    Sequential(
      (0): Dense(None -> 10, Activation(relu))
      (1): Dense(None -> 4, linear)
    )

Or use ``nn.Block`` to construct a network in a more flexible way.

.. doctest::

    >>> from mxnet import nd
    >>> from mxnet.gluon import nn
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

More tutorials are available at XXX.

.. currentmodule:: mxnet.gluon

.. autosummary::
    :toctree: out_construct
    :nosignatures:

    nn.Block
    nn.Sequential
    contrib.nn.Concurrent

The hybridiziable containers allow to hybridize a network to enjoy a better
performance and to be deployed without Python later.

.. doctest::

    >>> from mxnet.gluon import nn
    >>> net = nn.HybridSequential()
    >>> net.add(nn.Dense(10, activation='relu'), nn.Dense(4))
    >>> net.hybridize()
    >>> print(net)
    HybridSequential(
      (0): Dense(None -> 10, Activation(relu))
      (1): Dense(None -> 4, linear)
    )

Refer to XXX to explain hybridize.

.. autosummary::
    :toctree: out_construct
    :nosignatures:

    nn.HybridBlock
    nn.HybridSequential
    contrib.nn.HybridConcurrent


TODO, cannot add ``nn.SymbolBlock`` because it fails the doctest. Don't know how
to place ``nn.Identity``
