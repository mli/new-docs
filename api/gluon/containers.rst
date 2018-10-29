Containers
==========

This section covers the classes to construct neural network models.

.. currentmodule:: mxnet.gluon.nn

Blocks
------

The :py:mod:`mxnet.gluon.nn` module provides three classes to construct basc
blocks for a neural network model:

.. autosummary::
   :nosignatures:

   Block
   HybridBlock
   SymbolBlock

.. toctree::
   :hidden:

   Block
   HybridBlock
   SymbolBlock

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

.. disqus::
