Containers
==========

Sequential containers
---------------------

Besides inheriting :py:class:`mxnet.gluon.nn.Block` to create a neural network
models, :py:mod:`mxnet.gluon.nn` provides two classes to construct a model by
stacking layers sequentially. Refer to XXX for tutorials how to use them.

.. currentmodule:: mxnet.gluon.nn

.. autosummary::
    :toctree: out_container
    :nosignatures:

    nn.Sequential
    nn.HybridSequential

Concurrent containers
---------------------

The :py:mod:`mxnet.gluon.contrib.nn` package provides two additional containers
to construct models with more than one path, such as the Residual block in
ResNet and Inception block in GoogLeNet.


.. currentmodule:: mxnet.gluon.contrib.nn
.. autosummary::
    :toctree: out_construct
    :nosignatures:

    contrib.nn.Concurrent
    contrib.nn.HybridConcurrent
