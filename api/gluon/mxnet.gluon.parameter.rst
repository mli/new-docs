``parameter``
=============

.. currentmodule:: mxnet.gluon


.. autosummary::
   :toctree: .

   Parameter
   Constant
   ParameterDict

where

- :py:class:`Parameter` is the class that wraps a learnable parameter, such as weight or bias.
- :py:class:`Constant` a subclass of :py:class:`Parameter` whose values will not be updated during training.
- :py:class:`ParameterDict` a dictionary of :py:class:`Parameter` instances, which is used to present all parameters in a neural network.

Check tutorial XXX.
