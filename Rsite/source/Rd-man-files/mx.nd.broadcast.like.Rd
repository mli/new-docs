% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mxnet_generated.R
\name{mx.nd.broadcast.like}
\alias{mx.nd.broadcast.like}
\title{Broadcasts lhs to have the same shape as rhs.}
\arguments{
\item{lhs}{NDArray-or-Symbol
First input.}

\item{rhs}{NDArray-or-Symbol
Second input.}

\item{lhs.axes}{Shape or None, optional, default=None
Axes to perform broadcast on in the first input array}

\item{rhs.axes}{Shape or None, optional, default=None
Axes to copy from the second input array}
}
\value{
out The result mx.ndarray
}
\description{
Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
with arrays of different shapes efficiently without creating multiple copies of arrays.
Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.
}
\details{
Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

For example::

broadcast_like([[1,2,3]], [[5,6,7],[7,8,9]]) = [[ 1.,  2.,  3.],
                                                   [ 1.,  2.,  3.]])

broadcast_like([9], [1,2,3,4,5], lhs_axes=(0,), rhs_axes=(-1,)) = [9,9,9,9,9]



Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L315
}

