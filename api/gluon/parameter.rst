Parameters
==========

Gluon handles parameters in a nerual network layer through module:

.. autosummary::
   :toctree: out_param

   mxnet.gluon.parameter


This module provides three classes:

.. currentmodule:: mxnet.gluon.parameter

.. autosummary::
    :toctree: out_param
    :nosignatures:

    Parameter
    Constant
    ParameterDict

where

- :py:class:`Parameter` is the class that wraps a learnable parameter, such as weight or bias.
- :py:class:`Constant` a subclass of :py:class:`Parameter` whose values will not be updated during training.
- :py:class:`ParameterDict` a dictionary of :py:class:`Parameter` instances, which is used to present all parameters in a neural network.

Check tutorial XXX.

:py:class:`Parameter` class
---------------------------

.. autosummary::
    :toctree: out_param
    :nosignatures:

    Parameter.cast
    Parameter.data
    Parameter.grad
    Parameter.initialize
    Parameter.list_ctx
    Parameter.list_data
    Parameter.list_grad
    Parameter.list_row_sparse_data
    Parameter.reset_ctx
    Parameter.row_sparse_data
    Parameter.set_data
    Parameter.var
    Parameter.zero_grad

:py:class:`ParameterDict` class
--------------------------------

.. autosummary::
    :toctree: out_param
    :nosignatures:

    ParameterDict.get
    ParameterDict.get_constant
    ParameterDict.initialize
    ParameterDict.items
    ParameterDict.keys
    ParameterDict.load
    ParameterDict.reset_ctx
    ParameterDict.save
    ParameterDict.setattr
    ParameterDict.update
    ParameterDict.values
    ParameterDict.zero_grad
