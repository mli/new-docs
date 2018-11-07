``class Symbol``
==================

Composition
------------------

Composite multiple symbols into a new one by an operator.


.. autosummary::
    :nosignatures:

    Symbol.__call__


Arithmetic operations
------------------------


.. autosummary::
    :nosignatures:

    Symbol.__add__
    Symbol.__sub__
    Symbol.__rsub__
    Symbol.__neg__
    Symbol.__mul__
    Symbol.__div__
    Symbol.__rdiv__
    Symbol.__mod__
    Symbol.__rmod__
    Symbol.__pow__


Trigonometric functions
------------------------


.. autosummary::
    :nosignatures:

    Symbol.sin
    Symbol.cos
    Symbol.tan
    Symbol.arcsin
    Symbol.arccos
    Symbol.arctan
    Symbol.degrees
    Symbol.radians


Hyperbolic functions
------------------------


.. autosummary::
    :nosignatures:

    Symbol.sinh
    Symbol.cosh
    Symbol.tanh
    Symbol.arcsinh
    Symbol.arccosh
    Symbol.arctanh


Exponents and logarithms
------------------------


.. autosummary::
    :nosignatures:

    Symbol.exp
    Symbol.expm1
    Symbol.log
    Symbol.log10
    Symbol.log2
    Symbol.log1p


Powers
------------------------


.. autosummary::
    :nosignatures:

    Symbol.sqrt
    Symbol.rsqrt
    Symbol.cbrt
    Symbol.rcbrt
    Symbol.square


Basic neural network functions
----------------------------------


.. autosummary::
    :nosignatures:

    Symbol.relu
    Symbol.sigmoid
    Symbol.softmax
    Symbol.log_softmax


Comparison operators
----------------------


.. autosummary::
    :nosignatures:

    Symbol.__lt__
    Symbol.__le__
    Symbol.__gt__
    Symbol.__ge__
    Symbol.__eq__
    Symbol.__ne__


Symbol creation
---------------------


.. autosummary::
    :nosignatures:

    Symbol.zeros_like
    Symbol.ones_like
    Symbol.diag


Changing shape and type
---------------------------


.. autosummary::
    :nosignatures:

    Symbol.astype
    Symbol.shape_array
    Symbol.size_array
    Symbol.reshape
    Symbol.reshape_like
    Symbol.flatten
    Symbol.expand_dims


Expanding elements
-----------------------


.. autosummary::
    :nosignatures:

    Symbol.broadcast_to
    Symbol.broadcast_axes
    Symbol.broadcast_like
    Symbol.tile
    Symbol.pad


Rearranging elements
----------------------


.. autosummary::
    :nosignatures:

    Symbol.transpose
    Symbol.swapaxes
    Symbol.flip
    Symbol.depth_to_space
    Symbol.space_to_depth


Reduce functions
---------------------------


.. autosummary::
    :nosignatures:

    Symbol.sum
    Symbol.nansum
    Symbol.prod
    Symbol.nanprod
    Symbol.mean
    Symbol.max
    Symbol.min
    Symbol.norm


Rounding
---------------------


.. autosummary::
    :nosignatures:

    Symbol.round
    Symbol.rint
    Symbol.fix
    Symbol.floor
    Symbol.ceil
    Symbol.trunc


Sorting and searching
-----------------------------


.. autosummary::
    :nosignatures:

    Symbol.sort
    Symbol.argsort
    Symbol.topk
    Symbol.argmax
    Symbol.argmin
    Symbol.argmax_channel


Query information
--------------------


.. autosummary::
    :nosignatures:

    Symbol.name
    Symbol.list_arguments
    Symbol.list_outputs
    Symbol.list_auxiliary_states
    Symbol.list_attr
    Symbol.attr
    Symbol.attr_dict


Indexing
-----------------------


.. autosummary::
    :nosignatures:

    Symbol.slice
    Symbol.slice_axis
    Symbol.slice_like
    Symbol.take
    Symbol.one_hot
    Symbol.pick
    Symbol.ravel_multi_index
    Symbol.unravel_index


Get internal and output symbol
----------------------------------


.. autosummary::
    :nosignatures:

    Symbol.__getitem__
    Symbol.__iter__
    Symbol.get_internals
    Symbol.get_children


Inference type and shape
----------------------------------


.. autosummary::
    :nosignatures:

    Symbol.infer_type
    Symbol.infer_shape
    Symbol.infer_shape_partial



Bind
------------------


.. autosummary::
    :nosignatures:

    Symbol.bind
    Symbol.simple_bind


Save
------------------


.. autosummary::
    :nosignatures:

    Symbol.save
    Symbol.tojson
    Symbol.debug_str


Miscellaneous
-----------------------


.. autosummary::
    :nosignatures:

    Symbol.clip
    Symbol.sign
