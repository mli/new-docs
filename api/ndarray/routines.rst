Routines in ``mxnet.ndarray``
====================================

Create Arrays
-------------

.. currentmodule:: mxnet.ndarray

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    array
    empty
    zeros
    zeros_like
    ones
    ones_like
    full
    arange
    diag
    load
    save

Manipulate
------------


Change shape and type
^^^^^^^^^^^^^^^^^^^^^^^


.. autosummary::
    :nosignatures:
    :toctree: _autogen

    cast
    shape_array
    size_array
    reshape
    reshape_like
    flatten
    expand_dims

Expand  elements
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    broadcast_to
    broadcast_axes
    broadcast_like
    repeat
    tile
    pad

Rearrange elements
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    transpose
    swapaxes
    flip
    depth_to_space
    space_to_depth

Join and split
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    concat
    split
    stack

Index
^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    slice
    slice_axis
    slice_like
    take
    batch_take
    one_hot
    pick
    where
    ravel_multi_index
    unravel_index


Sequence
^^^^^^^^

.. currentmodule:: mxnet.ndarray

.. autosummary::
    :toctree: _autogen

    SequenceLast
    SequenceMask
    SequenceReverse

Math
----

Arithmetic
^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    add
    subtract
    negative
    multiply
    divide
    modulo
    dot
    batch_dot
    add_n

Trigonometric
^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sin
    cos
    tan
    arcsin
    arccos
    arctan
    broadcast_hypot
    degrees
    radians

Hyperbolic
^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sinh
    cosh
    tanh
    arcsinh
    arccosh
    arctanh

Reduce
^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sum
    nansum
    prod
    nanprod
    mean
    max
    min
    norm

Round
^^^^^^^^

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

    exp
    expm1
    log
    log10
    log2
    log1p

Powers
^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    power
    sqrt
    rsqrt
    cbrt
    rcbrt
    square
    reciprocal

Compare
^^^^^^^^^^


.. autosummary::
    :nosignatures:
    :toctree: _autogen

    equal
    not_equal
    greater
    greater_equal
    lesser
    lesser_equal

Logical
^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    logical_and
    logical_or
    logical_xor
    logical_not

Sort and Search
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autogen

    sort
    topk
    argsort
    argmax
    argmin

Random Distribution
^^^^^^^^^^^^^^^^^^^^


.. autosummary::
    :nosignatures:
    :toctree: _autogen

    random.randn
    random.exponential
    random.gamma
    random.generalized_negative_binomial
    random.negative_binomial
    random.normal
    random.poisson
    random.uniform
    random.multinomial
    random.shuffle

Linear Algebra
^^^^^^^^^^^^^^


.. autosummary::
    :nosignatures:
    :toctree: _autogen

    linalg.gemm
    linalg.gemm2
    linalg.potrf
    linalg.potri
    linalg.trmm
    linalg.trsm
    linalg.sumlogdiag
    linalg.syrk
    linalg.gelqf
    linalg.syevd


Miscellaneous
-------------

.. autosummary::
    :toctree: _autogen

    maximum
    minimum
    clip
    abs
    sign
    gamma
    gammaln

Neural Network
--------------

.. autosummary::
    :toctree: _autogen

    FullyConnected
    Convolution
    Activation
    BatchNorm
    Pooling
    SoftmaxOutput
    softmax
    log_softmax
    relu
    sigmoid
    Correlation
    Deconvolution
    RNN
    Embedding
    LeakyReLU
    InstanceNorm
    LayerNorm
    L2Normalization
    LRN
    ROIPooling
    SoftmaxActivation
    Dropout
    BilinearSampler
    GridGenerator
    UpSampling
    SpatialTransformer
    LinearRegressionOutput
    LogisticRegressionOutput
    MAERegressionOutput
    SVMOutput
    softmax_cross_entropy
    smooth_l1
    IdentityAttachKLSparseReg
    MakeLoss
    BlockGrad
    Custom
