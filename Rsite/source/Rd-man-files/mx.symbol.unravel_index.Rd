% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mxnet_generated.R
\name{mx.symbol.unravel_index}
\alias{mx.symbol.unravel_index}
\title{unravel_index:Converts an array of flat indices into a batch of index arrays. The operator follows numpy conventions so a single multi index is given by a column of the output matrix.}
\usage{
mx.symbol.unravel_index(...)
}
\arguments{
\item{data}{NDArray-or-Symbol
Array of flat indices}

\item{shape}{Shape(tuple), optional, default=[]
Shape of the array into which the multi-indices apply.}

\item{name}{string, optional
Name of the resulting symbol.}
}
\value{
out The result mx.symbol
}
\description{
Examples::
}
\details{
A = [22,41,37]
   unravel(A, shape=(7,6)) = [[3,6,6],[4,5,1]]



Defined in src/operator/tensor/ravel.cc:L65
}

