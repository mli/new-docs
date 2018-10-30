Layers
======

Gluon provides a large number of build-in neural network layers in the following
modules


.. autosummary::
    :nosignatures:

    mxnet.gluon.nn
    mxnet.gluon.rnn
    mxnet.gluon.contrib.nn
    mxnet.gluon.contrib.rnn

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

Basic Layers
------------

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    nn.Dense
    nn.Activation
    nn.Dropout
    nn.Flatten
    nn.Lambda
    nn.HybridLambda

Convolutional Layers
--------------------

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    nn.Conv1D
    nn.Conv2D
    nn.Conv3D
    nn.Conv1DTranspose
    nn.Conv2DTranspose
    nn.Conv3DTranspose

Pooling Layers
--------------

.. autosummary::
   :toctree: _autogen
    :nosignatures:

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
    :toctree: _autogen
    :nosignatures:

    nn.BatchNorm
    nn.InstanceNorm
    nn.LayerNorm
    contrib.nn.SyncBatchNorm

Embedding Layers
----------------

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    nn.Embedding
    contrib.nn.SparseEmbedding

Recurrent Cells
----------------

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    rnn.LSTMCell
    rnn.GRUCell
    rnn.RecurrentCell
    rnn.SequentialRNNCell
    rnn.BidirectionalCell
    rnn.DropoutCell
    rnn.ZoneoutCell
    rnn.ResidualCell
    contrib.rnn.Conv1DRNNCell
    contrib.rnn.Conv2DRNNCell
    contrib.rnn.Conv3DRNNCell
    contrib.rnn.Conv1DLSTMCell
    contrib.rnn.Conv2DLSTMCell
    contrib.rnn.Conv3DLSTMCell
    contrib.rnn.Conv1DGRUCell
    contrib.rnn.Conv2DGRUCell
    contrib.rnn.Conv3DGRUCell
    contrib.rnn.VariationalDropoutCell
    contrib.rnn.LSTMPCell

Recurrent Layers
----------------

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    rnn.RNN
    rnn.LSTM
    rnn.GRU

Advanced Activation Layers
--------------------------

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    nn.LeakyReLU
    nn.PReLU
    nn.ELU
    nn.SELU
    nn.Swish
