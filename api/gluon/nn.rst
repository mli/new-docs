``nn`` and ``contrib.nn``
=========================

Gluon provides a large number of build-in neural network layers in the following
modules

.. autosummary::
    :nosignatures:

    mxnet.gluon.nn
    mxnet.gluon.contrib.nn

The following codes shows a basic usage of a layer.

.. doctest::

   >>> from mxnet import nd
   >>> from mxnet.gluon import nn
   >>> layer = nn.Dense(2)  # construct a dense layers with 2 output units
   >>> layer.initialize()  # intialize its parameters
   >>> x = nd.random.uniform(shape=(4, 10))
   >>> print(layer(x).shape)  # run a forward propogation and print output shape
   (4, 2)
   >>> print(layer.weight)  # get its weight parameters
   Parameter dense3_weight (shape=(2, 10), dtype=float32)

Refer to :doc:`../../develop/tutorials/index.rst` for how to use them to
construct various neural networks

In the reset of this secion, we group all build-in layers according to their
categories.

.. currentmodule:: mxnet.gluon

Blocks
------

The :py:mod:`mxnet.gluon.nn` module provides three classes to construct basc
blocks for a neural network model:

.. autosummary::
   :nosignatures:
   :toctree: .

   nn.Block
   nn.HybridBlock
   nn.SymbolBlock

The difference between these three classes:

- :py:class:`Block`: the bass class for any neural nework layers and models.
- :py:class:`HybridBlock`: a subclass of :py:class:`Block` that allows to
  hybridize a model. It constraints operations can be run in the ``forward``
  method, e.g. the `print` function doesn't work any more. Check tutorial XXX
  for more details.

- :py:class:`SymbolBlock`: a sublcass of :py:class:`Block` that is able to wrap
  a :py:class:`mxnet.symbol.Symbol` instance into a :py:class:`Block`
  instance. Check XXX-Symbol tutorials and how to XXX to use this class.


Sequential containers
---------------------

Besides inheriting :py:class:`mxnet.gluon.nn.Block` to create a neural network
models, :py:mod:`mxnet.gluon.nn` provides two classes to construct a model by
stacking layers sequentially. Refer to XXX for tutorials how to use them.


.. autosummary::
    :toctree: _autogen
    :nosignatures:

    nn.Sequential
    nn.HybridSequential

Concurrent containers
---------------------

The :py:mod:`mxnet.gluon.contrib.nn` package provides two additional containers
to construct models with more than one path, such as the Residual block in
ResNet and Inception block in GoogLeNet.


.. autosummary::
    :toctree: _autogen
    :nosignatures:

    contrib.nn.Concurrent
    contrib.nn.HybridConcurrent


Basic Layers
------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.Dense
    nn.Activation
    nn.Dropout
    nn.Flatten
    nn.Lambda
    nn.HybridLambda

Convolutional Layers
--------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.Conv1D
    nn.Conv2D
    nn.Conv3D
    nn.Conv1DTranspose
    nn.Conv2DTranspose
    nn.Conv3DTranspose

Pooling Layers
--------------

.. autosummary::
   :nosignatures:
   :toctree: _autogen

    nn.MaxPool1D
    nn.MaxPool2D
    nn.MaxPool3D
    nn.AvgPool1D
    nn.AvgPool2D
    nn.AvgPool3D
    nn.GlobalMaxPool1D
    nn.GlobalMaxPool2D
    nn.GlobalMaxPool3D
    nn.GlobalAvgPool1D
    nn.GlobalAvgPool2D
    nn.GlobalAvgPool3D
    nn.ReflectionPad2D

Normalization Layers
--------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.BatchNorm
    nn.InstanceNorm
    nn.LayerNorm
    contrib.nn.SyncBatchNorm

Embedding Layers
----------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.Embedding
    contrib.nn.SparseEmbedding


Advanced Activation Layers
--------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.LeakyReLU
    nn.PReLU
    nn.ELU
    nn.SELU
    nn.Swish
