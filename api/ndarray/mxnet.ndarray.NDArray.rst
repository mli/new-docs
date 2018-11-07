NDArray
=================

.. currentmodule:: mxnet.ndarray

.. autoclass:: NDArray

Attributes
----------

.. autosummary::
   :toctree: _autogen

   NDArray.shape
   NDArray.ndim
   NDArray.size
   NDArray.context
   NDArray.dtype
   NDArray.stype
   NDArray.grad
   NDArray.handle
   NDArray.writable

Array creation
--------------

.. autosummary::
   :toctree: _autogen

   NDArray.zeros_like
   NDArray.ones_like

Manipulation
-------------

Array conversion
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.copy
   NDArray.copyto
   NDArray.as_in_context
   NDArray.asnumpy
   NDArray.asscalar
   NDArray.astype
   NDArray.tostype

Changing shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.shape_array
   NDArray.size_array
   NDArray.reshape
   NDArray.reshape_like
   NDArray.flatten
   NDArray.expand_dims
   NDArray.split
   NDArray.diag

Expanding elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.broadcast_to
   NDArray.broadcast_axes
   NDArray.broadcast_like
   NDArray.repeat
   NDArray.tile
   NDArray.pad

Rearrange elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.T
   NDArray.transpose
   NDArray.swapaxes
   NDArray.flip
   NDArray.depth_to_space
   NDArray.space_to_depth


Sorting and searching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.sort
   NDArray.argsort
   NDArray.topk
   NDArray.argmin
   NDArray.argmax
   NDArray.argmax_channel

Indexing
^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__getitem__
   NDArray.__setitem__
   NDArray.slice
   NDArray.slice_axis
   NDArray.slice_like
   NDArray.take
   NDArray.one_hot
   NDArray.pick

Lazy evaluation
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.wait_to_read

Math
----

Arithmetic operations
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__add__
   NDArray.__sub__
   NDArray.__rsub__
   NDArray.__neg__
   NDArray.__mul__
   NDArray.__div__
   NDArray.__rdiv__
   NDArray.__mod__
   NDArray.__rmod__
   NDArray.__pow__

Rounding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.round
   NDArray.rint
   NDArray.fix
   NDArray.floor
   NDArray.ceil
   NDArray.trunc

Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.sum
   NDArray.nansum
   NDArray.prod
   NDArray.nanprod
   NDArray.mean
   NDArray.max
   NDArray.min
   NDArray.norm

In^place arithmetic operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__iadd__
   NDArray.__isub__
   NDArray.__imul__
   NDArray.__idiv__
   NDArray.__imod__

Comparison operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__lt__
   NDArray.__le__
   NDArray.__gt__
   NDArray.__ge__
   NDArray.__eq__
   NDArray.__ne__

Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.sin
   NDArray.cos
   NDArray.tan
   NDArray.arccos
   NDArray.arcsin
   NDArray.arctan
   NDArray.degrees
   NDArray.radians

Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.sinh
   NDArray.cosh
   NDArray.arccosh
   NDArray.arcsinh
   NDArray.arctanh
   NDArray.tanh

Exponents and logarithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.exp
   NDArray.expm1
   NDArray.log
   NDArray.log2
   NDArray.log10
   NDArray.log1p

Powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.sqrt
   NDArray.rsqrt
   NDArray.cbrt
   NDArray.rcbrt
   NDArray.square
   NDArray.reciprocal

Miscellaneous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.clip
   NDArray.sign

Neural network
------------------

.. autosummary::
   :toctree: _autogen

   NDArray.relu
   NDArray.sigmoid
   NDArray.softmax
   NDArray.log_softmax




.. disqus::
   :disqus_identifier: mxnet.ndarray.NDArray
