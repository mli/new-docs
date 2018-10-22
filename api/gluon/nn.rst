Gluon Neural Network Layers
===========================

This document lists the neural network blocks in Gluon:

.. currentmodule:: mxnet.gluon.nn


Basic Layers
------------


.. autosummary::
    :nosignatures:

    Dense
    Dropout
    BatchNorm
    InstanceNorm
    LayerNorm
    Embedding
    Flatten
    Lambda
    HybridLambda


Convolutional Layers
--------------------


.. autosummary::
    :nosignatures:

    Conv1D
    Conv2D
    Conv3D
    Conv1DTranspose
    Conv2DTranspose
    Conv3DTranspose



Pooling Layers
--------------


.. autosummary::
    :nosignatures:

    MaxPool1D
    MaxPool2D
    MaxPool3D
    AvgPool1D
    AvgPool2D
    AvgPool3D
    GlobalMaxPool1D
    GlobalMaxPool2D
    GlobalMaxPool3D
    GlobalAvgPool1D
    GlobalAvgPool2D
    GlobalAvgPool3D
    ReflectionPad2D

Activation Layers
-----------------


.. autosummary::
    :nosignatures:

    Activation
    LeakyReLU
    PReLU
    ELU
    SELU
    Swish


API Reference
------------

.. automodule:: mxnet.gluon.nn
    :members:
    :imported-members:
    :exclude-members: Block, HybridBlock, SymbolBlock, Sequential, HybridSequential
