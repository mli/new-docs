Layers
======

Overview
--------

A layer is xxx.

.. currentmodule:: mxnet.gluon

Basic Layers
------------


.. autosummary::
    :toctree: out_basic
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
    :toctree: out_conv
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
    :toctree: out_pool
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
    :toctree: out_normal
    :nosignatures:

    nn.BatchNorm
    nn.InstanceNorm
    nn.LayerNorm
    contrib.nn.SyncBatchNorm

Embedding Layers
----------------

.. autosummary::
    :toctree: out_embed
    :nosignatures:

    nn.Embedding
    contrib.nn.SparseEmbedding

Recurrent Cells
----------------

.. autosummary::
    :toctree: out_rnn_cell
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
    :toctree: out_rnn_layer
    :nosignatures:

    rnn.RNN
    rnn.LSTM
    rnn.GRU

Advanced Activation Layers
--------------------------

.. autosummary::
    :toctree: out_act
    :nosignatures:

    nn.LeakyReLU
    nn.PReLU
    nn.ELU
    nn.SELU
    nn.Swish
