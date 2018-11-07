Routines in ``mxnet.ndarray.sparse``
=====================================

.. currentmodule:: mxnet.ndarray.sparse

Create Arrays
--------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    array
    empty
    zeros
    zeros_like
    csr_matrix
    row_sparse_array

Manipulate
------------


Change shape and type
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    cast_storage

Index
^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    slice
    retain
    where

Math
----

Arithmetic
^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    elemwise_add
    elemwise_sub
    elemwise_mul
    broadcast_add
    broadcast_sub
    broadcast_mul
    broadcast_div
    negative
    dot
    add_n

Trigonometric
^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sin
    tan
    arcsin
    arctan
    degrees
    radians

Hyperbolic
^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sinh
    tanh
    arcsinh
    arctanh

Reduce
^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sum
    mean
    norm

Round
^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    round
    rint
    fix
    floor
    ceil
    trunc

Exponents and logarithms
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    expm1
    log1p

Powers
^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sqrt
    square

Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    abs
    sign

Neural network
---------------

Updater
^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sgd_update
    sgd_mom_update
    adam_update
    adagrad_update

More
^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    make_loss
    stop_gradient
    Embedding
    LinearRegressionOutput
    LogisticRegressionOutput
