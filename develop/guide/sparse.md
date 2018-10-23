# Sparse NDArray

## RowSparseNDArray - NDArray for Sparse Gradient Updates

### Motivation

Many real world datasets deal with high dimensional sparse feature vectors. When learning
the weights of models with sparse datasets, the derived gradients of the weights could be sparse.

Let's say we perform a matrix multiplication of ``X``  and ``W``, where ``X`` is a 1x2 matrix, and ``W`` is a 2x3 matrix. Let ``Y`` be the matrix multiplication of the two matrices:


```python
import mxnet as mx
X = mx.nd.array([[1,0]])
W = mx.nd.array([[3,4,5], [6,7,8]])
Y = mx.nd.dot(X, W)
{'X': X, 'W': W, 'Y': Y}
```

As you can see,

```bash
Y[0][0] = X[0][0] * W[0][0] + X[0][1] * W[1][0] = 1 * 3 + 0 * 6 = 3
Y[0][1] = X[0][0] * W[0][1] + X[0][1] * W[1][1] = 1 * 4 + 0 * 7 = 4
Y[0][2] = X[0][0] * W[0][2] + X[0][1] * W[1][2] = 1 * 5 + 0 * 8 = 5
```

What about dY / dW, the gradient for ``W``? Let's call it ``grad_W``. To start with, the shape of ``grad_W`` is the same as that of ``W`` as we are taking the derivatives with respect to ``W``, which is 2x3. Then we calculate each entry in ``grad_W`` as follows:

```bash
grad_W[0][0] = X[0][0] = 1
grad_W[0][1] = X[0][0] = 1
grad_W[0][2] = X[0][0] = 1
grad_W[1][0] = X[0][1] = 0
grad_W[1][1] = X[0][1] = 0
grad_W[1][2] = X[0][1] = 0
```

As a matter of fact, you can calculate ``grad_W`` by multiplying the transpose of ``X`` with a matrix of ones:


```python
grad_W = mx.nd.dot(X, mx.nd.ones_like(Y), transpose_a=True)
grad_W
```

As you can see, row 0 of ``grad_W`` contains non-zero values while row 1 of ``grad_W`` does not. Why did that happen?
If you look at how ``grad_W`` is calculated, notice that since column 1 of ``X`` is filled with zeros, row 1 of ``grad_W`` is filled with zeros too.

In the real world, gradients for parameters that interact with sparse inputs ususally have gradients where many row slices are completely zeros. Storing and manipulating such sparse matrices with many row slices of all zeros in the default dense structure results in wasted memory and processing on the zeros. More importantly, many gradient based optimization methods such as SGD, [AdaGrad](https://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf) and [Adam](https://arxiv.org/pdf/1412.6980.pdf)
take advantage of sparse gradients and prove to be efficient and effective.
**In MXNet, the ``RowSparseNDArray`` stores the matrix in ``row sparse`` format, which is designed for arrays of which most row slices are all zeros.**
In this tutorial, we will describe what the row sparse format is and how to use RowSparseNDArray for sparse gradient updates in MXNet.

### Row Sparse Format

A RowSparseNDArray represents a multidimensional NDArray of shape `[LARGE0, D1, .. , Dn]` using two separate 1D arrays:
`data` and `indices`.

- data: an NDArray of any dtype with shape `[D0, D1, ..., Dn]`.
- indices: a 1D int64 NDArray with shape `[D0]` with values sorted in ascending order.

The ``indices`` array stores the indices of the row slices with **non-zeros**,
while the values are stored in ``data`` array. The corresponding NDArray `dense` represented by RowSparseNDArray `rsp` has

``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

A RowSparseNDArray is typically used to represent non-zero row slices of a large NDArray of shape `[LARGE0, D1, .. , Dn]` where LARGE0 >> D0 and most row slices are zeros.

Given this two-dimension matrix:


```python
[[ 1, 2, 3],
 [ 0, 0, 0],
 [ 4, 0, 5],
 [ 0, 0, 0],
 [ 0, 0, 0]]
```

The row sparse representation would be:
- `data` array holds all the non-zero row slices of the array.
- `indices` array stores the row index for each row slice with non-zero elements.


```python
data = [[1, 2, 3], [4, 0, 5]]
indices = [0, 2]
```

`RowSparseNDArray` supports multidimensional arrays. Given this 3D tensor:

```python
[[[1, 0],
  [0, 2],
  [3, 4]],

 [[5, 0],
  [6, 0],
  [0, 0]],

 [[0, 0],
  [0, 0],
  [0, 0]]]
```

The row sparse representation would be (with `data` and `indices` defined the same as above):


```python
data = [[[1, 0], [0, 2], [3, 4]], [[5, 0], [6, 0], [0, 0]]]
indices = [0, 1]
```

``RowSparseNDArray`` is a subclass of ``NDArray``. If you query **stype** of a RowSparseNDArray,
the value will be **"row_sparse"**.

### Array Creation

You can create a `RowSparseNDArray` with data and indices by using the `row_sparse_array` function:


```python
import mxnet as mx
import numpy as np
# Create a RowSparseNDArray with python lists
shape = (6, 2)
data_list = [[1, 2], [3, 4]]
indices_list = [1, 4]
a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
# Create a RowSparseNDArray with numpy arrays
data_np = np.array([[1, 2], [3, 4]])
indices_np = np.array([1, 4])
b = mx.nd.sparse.row_sparse_array((data_np, indices_np), shape=shape)
{'a':a, 'b':b}
```



### Function Overview

Similar to `CSRNDArray`, the are several functions with `RowSparseNDArray` that behave the same way. In the code blocks below you can try out these common functions:

- **.dtype** - to set the data type
- **.asnumpy** - to cast as a numpy array for inspecting it
- **.data** - to access the data array
- **.indices** - to access the indices array
- **.tostype** - to set the storage type
- **.cast_storage** - to convert the storage type
- **.copy** - to copy the array
- **.copyto** - to copy to deep copy an existing array


### Setting Type

You can create a `RowSparseNDArray` from another specifying the element data type with the option `dtype`, which accepts a numpy type. By default, `float32` is used.


```python
# Float32 is used by default
c = mx.nd.sparse.array(a)
# Create a 16-bit float array
d = mx.nd.array(a, dtype=np.float16)
(c.dtype, d.dtype)
```







### Inspecting Arrays

As with `CSRNDArray`, you can inspect the contents of a `RowSparseNDArray` by filling
its contents into a dense `numpy.ndarray` using the `asnumpy` function.


```python
a.asnumpy()
```



You can inspect the internal storage of a RowSparseNDArray by accessing attributes such as `indices` and `data`:


```python
# Access data array
data = a.data
# Access indices array
indices = a.indices
{'a.stype': a.stype, 'data':data, 'indices':indices}
```



### Storage Type Conversion

You can convert an NDArray to a RowSparseNDArray and vice versa by using the `tostype` function:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type from `default` to `row_sparse`
rsp = ones.tostype('row_sparse')
# Cast the storage type from `row_sparse` to `default`
dense = rsp.tostype('default')
{'rsp':rsp, 'dense':dense}
```

You can also convert the storage type by using the `cast_storage` operator:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type to `row_sparse`
rsp = mx.nd.sparse.cast_storage(ones, 'row_sparse')
# Cast the storage type to `default`
dense = mx.nd.sparse.cast_storage(rsp, 'default')
{'rsp':rsp, 'dense':dense}
```




### Copies

You can use the `copy` method which makes a deep copy of the array and its data, and returns a new array.
We can also use the `copyto` method or the slice operator `[]` to deep copy to an existing array.


```python
a = mx.nd.ones((2,2)).tostype('row_sparse')
b = a.copy()
c = mx.nd.sparse.zeros('row_sparse', (2,2))
c[:] = a
d = mx.nd.sparse.zeros('row_sparse', (2,2))
a.copyto(d)
{'b is a': b is a, 'b.asnumpy()':b.asnumpy(), 'c.asnumpy()':c.asnumpy(), 'd.asnumpy()':d.asnumpy()}
```




If the storage types of source array and destination array do not match,
the storage type of destination array will not change when copying with `copyto` or the slice operator `[]`. The source array will be temporarily converted to desired storage type before the copy.


```python
e = mx.nd.sparse.zeros('row_sparse', (2,2))
f = mx.nd.sparse.zeros('row_sparse', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype, 'g.stype':g.stype}
```







### Retain Row Slices

You can retain a subset of row slices from a RowSparseNDArray specified by their row indices.


```python
data = [[1, 2], [3, 4], [5, 6]]
indices = [0, 2, 3]
rsp = mx.nd.sparse.row_sparse_array((data, indices), shape=(5, 2))
# Retain row 0 and row 1
rsp_retained = mx.nd.sparse.retain(rsp, mx.nd.array([0, 1]))
{'rsp.asnumpy()': rsp.asnumpy(), 'rsp_retained': rsp_retained, 'rsp_retained.asnumpy()': rsp_retained.asnumpy()}
```




### Sparse Operators and Storage Type Inference

Operators that have specialized implementation for sparse arrays can be accessed in ``mx.nd.sparse``. You can read the [mxnet.ndarray.sparse API documentation](http://mxnet.incubator.apache.org/api/python/ndarray/sparse.html) to find what sparse operators are available.


```python
shape = (3, 5)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
# A csr matrix as lhs
lhs = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
# A dense matrix as rhs
rhs = mx.nd.ones((3, 2))
# row_sparse result is inferred from sparse operator dot(csr.T, dense) based on input stypes
transpose_dot = mx.nd.sparse.dot(lhs, rhs, transpose_a=True)
{'transpose_dot': transpose_dot, 'transpose_dot.asnumpy()': transpose_dot.asnumpy()}
```



For any sparse operator, the storage type of output array is inferred based on inputs. You can either read the documentation or inspect the `stype` attribute of output array to know what storage type is inferred:


```python
a = transpose_dot.copy()
b = a * 2  # b will be a RowSparseNDArray since zero multiplied by 2 is still zero
c = a + mx.nd.ones((5, 2))  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```







For operators that don't specialize in sparse arrays, you can still use them with sparse inputs with some performance penalty.
In MXNet, dense operators require all inputs and outputs to be in the dense format.

If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily so that the dense operator can be used.

If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format.

For operators that don't specialize in sparse arrays, you can still use them with sparse inputs with some performance penalty.


```python
e = mx.nd.sparse.zeros('row_sparse', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd.stype':d.stype, 'e.stype':e.stype} # stypes of a and e will be not changed
```







Note that warning messages will be printed when such a storage fallback event happens. If you are using jupyter notebook, the warning message will be printed in your terminal console.

### Sparse Optimizers

In MXNet, sparse gradient updates are applied when weight, state and gradient are all in `row_sparse` storage.
The sparse optimizers only update the row slices of the weight and the states whose indices appear
in `gradient.indices`. For example, the default update rule for SGD optimizer is:

```bash
rescaled_grad = learning_rate * rescale_grad * clip(grad, clip_gradient) + weight_decay * weight
state = momentum * state + rescaled_grad
weight = weight - state
```

However, with sparse gradient the SGD optimizer uses the following lazy update by default:

```bash
for row in grad.indices:
    rescaled_grad[row] = learning_rate * rescale_grad * clip(grad[row], clip_gradient) + weight_decay * weight[row]
    state[row] = momentum[row] * state[row] + rescaled_grad[row]
    weight[row] = weight[row] - state[row]
```

This means that the lazy update leads to different optimization results if `weight_decay` or `momentum` is non-zero.
To disable lazy update, please set `lazy_update` to be False when creating the optimizer.


```python
# Create weight
shape = (4, 2)
weight = mx.nd.ones(shape).tostype('row_sparse')
# Create gradient
data = [[1, 2], [4, 5]]
indices = [1, 2]
grad = mx.nd.sparse.row_sparse_array((data, indices), shape=shape)
sgd = mx.optimizer.SGD(learning_rate=0.01, momentum=0.01)
# Create momentum
momentum = sgd.create_state(0, weight)
# Before the update
{"grad.asnumpy()":grad.asnumpy(), "weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```



```python
sgd.update(0, weight, grad, momentum)
# Only row 0 and row 2 are updated for both weight and momentum
{"weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```





Note that only [mxnet.optimizer.SGD](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#mxnet.optimizer.SGD), [mxnet.optimizer.Adam](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#mxnet.optimizer.Adam), and
[mxnet.optimizer.AdaGrad](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#mxnet.optimizer.AdaGrad) support sparse updates in MXNet.

### Advanced Topics

#### GPU Support

By default, RowSparseNDArray operators are executed on CPU. To create a RowSparseNDArray on gpu, we need to explicitly specify the context:

**Note** If a GPU is not available, an error will be reported in the following section. In order to execute it on a cpu, set gpu_device to mx.cpu().


```python
import sys
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
try:
    a = mx.nd.sparse.zeros('row_sparse', (100, 100), ctx=gpu_device)
    a
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```

## CSRNDArray - NDArray in Compressed Sparse Row Storage Format

Many real world datasets deal with high dimensional sparse feature vectors. Take for instance a recommendation system where the number of categories and users is on the order of millions. The purchase data for each category by user would show that most users only make a few purchases, leading to a dataset with high sparsity (i.e. most of the elements are zeros).

Storing and manipulating such large sparse matrices in the default dense structure results in wasted memory and processing on the zeros. To take advantage of the sparse structure of the matrix, the `CSRNDArray` in MXNet stores the matrix in [compressed sparse row (CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) format and uses specialized algorithms in operators.
**The format is designed for 2D matrices with a large number of columns,
and each row is sparse (i.e. with only a few nonzeros).**

### Advantages of Compressed Sparse Row NDArray (CSRNDArray)
For matrices of high sparsity (e.g. ~1% non-zeros = ~1% density), there are two primary advantages of `CSRNDArray` over the existing `NDArray`:

- memory consumption is reduced significantly
- certain operations are much faster (e.g. matrix-vector multiplication)

You may be familiar with the CSR storage format in [SciPy](https://www.scipy.org/) and will note the similarities in MXNet's implementation. However there are some additional competitive features in `CSRNDArray` inherited from `NDArray`, such as non-blocking asynchronous evaluation and automatic parallelization that are not available in SciPy's flavor of CSR. You can find further explanations for evaluation and parallelization strategy in MXNet in the [NDArray tutorial](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html#lazy-evaluation-and-automatic-parallelization).

The introduction of `CSRNDArray` also brings a new attribute, `stype` as a holder for storage type info, to `NDArray`. You can query **ndarray.stype** now in addition to the oft-queried attributes such as **ndarray.shape**, **ndarray.dtype**, and **ndarray.context**. For a typical dense NDArray, the value of `stype` is **"default"**. For a `CSRNDArray`, the value of stype is **"csr"**.

### Compressed Sparse Row Matrix

A CSRNDArray represents a 2D matrix as three separate 1D arrays: **data**, **indptr** and **indices**, where the column indices for row `i` are stored in `indices[indptr[i]:indptr[i+1]]` in ascending order, and their corresponding values are stored in `data[indptr[i]:indptr[i+1]]`.

- **data**: CSR format data array of the matrix
- **indices**: CSR format index array of the matrix
- **indptr**: CSR format index pointer array of the matrix

#### Example Matrix Compression

For example, given the matrix:
```bash
[[7, 0, 8, 0]
 [0, 0, 0, 0]
 [0, 9, 0, 0]]
```

We can compress this matrix using CSR, and to do so we need to calculate `data`, `indices`, and `indptr`.

The `data` array holds all the non-zero entries of the matrix in row-major order. Put another way, you create a data array that has all of the zeros removed from the matrix, row by row, storing the numbers in that order. Your result:


```bash
data = [7, 8, 9]
```

The `indices` array stores the column index for each non-zero element in `data`. As you cycle through the data array, starting with 7, you can see it is in column 0. Then looking at 8, you can see it is in column 2. Lastly 9 is in column 1. Your result:


```bash
indices = [0, 2, 1]
```

The `indptr` array is what will help identify the rows where the data appears. It stores the offset into `data` of the first non-zero element number of each row of the matrix. This array always starts with 0 (reasons can be explored later), so indptr[0] is 0. Each subsequent value in the array is the aggregate number of non-zero elements up to that row. Looking at the first row of the matrix you can see two non-zero values, so indptr[1] is 2. The next row contains all zeros, so the aggregate is still 2, so indptr[2] is 2. Finally, you see the last row contains one non-zero element bring the aggregate to 3, so indptr[3] is 3. To reconstruct the dense matrix, you will use `data[0:2]` and `indices[0:2]` for the first row, `data[2:2]` and `indices[2:2]` for the second row (which contains all zeros), and `data[2:3]` and `indices[2:3]` for the third row. Your result:

```bash
indptr = [0, 2, 2, 3]
```


Note that in MXNet, the column indices for a given row are always sorted in ascending order,
and duplicated column indices for the same row are not allowed.

### Array Creation

There are a few different ways to create a `CSRNDArray`, but first let's recreate the matrix we just discussed using the `data`, `indices`, and `indptr` we calculated in the previous example.

You can create a CSRNDArray with data, indices and indptr by using the `csr_matrix` function:


```python
import mxnet as mx
# Create a CSRNDArray with python lists
shape = (3, 4)
data_list = [7, 8, 9]
indices_list = [0, 2, 1]
indptr_list = [0, 2, 2, 3]
a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
# Inspect the matrix
a.asnumpy()
```

```python
import numpy as np
# Create a CSRNDArray with numpy arrays
data_np = np.array([7, 8, 9])
indptr_np = np.array([0, 2, 2, 3])
indices_np = np.array([0, 2, 1])
b = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
b.asnumpy()
```

```python
# Compare the two. They are exactly the same.
{'a':a.asnumpy(), 'b':b.asnumpy()}
```

You can create an MXNet CSRNDArray from a `scipy.sparse.csr.csr_matrix` object by using the `array` function:


```python
try:
    import scipy.sparse as spsp
    # generate a csr matrix in scipy
    c = spsp.csr.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
    # create a CSRNDArray from a scipy csr object
    d = mx.nd.sparse.array(c)
    print('d:{}'.format(d.asnumpy()))
except ImportError:
    print("scipy package is required")
```

What if you have a big set of data and you haven't calculated indices or indptr yet? Let's try a simple CSRNDArray from an existing array of data and derive those values with some built-in functions. We can mockup a "big" dataset with a random amount of the data being non-zero, then compress it by using the `tostype` function, which is explained further in the [Storage Type Conversion](#storage-type-conversion) section:


```python
big_array = mx.nd.round(mx.nd.random.uniform(low=0, high=1, shape=(1000, 100)))
print(big_array)
big_array_csr = big_array.tostype('csr')
# Access indices array
indices = big_array_csr.indices
# Access indptr array
indptr = big_array_csr.indptr
# Access data array
data = big_array_csr.data
# The total size of `data`, `indices` and `indptr` arrays is much lesser than the dense big_array!
```

You can also create a CSRNDArray from another using the `array` function specifying the element data type with the option `dtype`,
which accepts a numpy type. By default, `float32` is used.


```python
# Float32 is used by default
e = mx.nd.sparse.array(a)
# Create a 16-bit float array
f = mx.nd.array(a, dtype=np.float16)
(e.dtype, f.dtype)
```



### Inspecting Arrays

A variety of methods are available for you to use for inspecting CSR arrays:
* **.asnumpy()**
* **.data**
* **.indices**
* **.indptr**

As you have seen already, we can inspect the contents of a `CSRNDArray` by filling
its contents into a dense `numpy.ndarray` using the `asnumpy` function.


```python
a.asnumpy()
```


You can also inspect the internal storage of a CSRNDArray by accessing attributes such as `indptr`, `indices` and `data`:


```python
# Access data array
data = a.data
# Access indices array
indices = a.indices
# Access indptr array
indptr = a.indptr
{'a.stype': a.stype, 'data':data, 'indices':indices, 'indptr':indptr}
```



### Storage Type Conversion

You can also convert storage types with:
* **tostype**
* **cast_storage**

To convert an NDArray to a CSRNDArray and vice versa by using the ``tostype`` function:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type from `default` to `csr`
csr = ones.tostype('csr')
# Cast the storage type from `csr` to `default`
dense = csr.tostype('default')
{'csr':csr, 'dense':dense}
```

To convert the storage type by using the `cast_storage` operator:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type to `csr`
csr = mx.nd.sparse.cast_storage(ones, 'csr')
# Cast the storage type to `default`
dense = mx.nd.sparse.cast_storage(csr, 'default')
{'csr':csr, 'dense':dense}
```

### Copies

You can use the `copy` method which makes a deep copy of the array and its data, and returns a new array.
You can also use the `copyto` method or the slice operator `[]` to deep copy to an existing array.


```python
a = mx.nd.ones((2,2)).tostype('csr')
b = a.copy()
c = mx.nd.sparse.zeros('csr', (2,2))
c[:] = a
d = mx.nd.sparse.zeros('csr', (2,2))
a.copyto(d)
{'b is a': b is a, 'b.asnumpy()':b.asnumpy(), 'c.asnumpy()':c.asnumpy(), 'd.asnumpy()':d.asnumpy()}
```

If the storage types of source array and destination array do not match,
the storage type of destination array will not change when copying with `copyto` or
the slice operator `[]`.


```python
e = mx.nd.sparse.zeros('csr', (2,2))
f = mx.nd.sparse.zeros('csr', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype, 'g.stype':g.stype}
```

### Indexing and Slicing
You can slice a CSRNDArray on axis 0 with operator `[]`, which copies the slices and returns a new CSRNDArray.


```python
a = mx.nd.array(np.arange(6).reshape(3,2)).tostype('csr')
b = a[1:2].asnumpy()
c = a[:].asnumpy()
{'a':a, 'b':b, 'c':c}
```

Note that multi-dimensional indexing or slicing along a particular axis is currently not supported for a CSRNDArray.

### Sparse Operators and Storage Type Inference

Operators that have specialized implementation for sparse arrays can be accessed in `mx.nd.sparse`. You can read the [mxnet.ndarray.sparse API documentation](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/sparse.html) to find what sparse operators are available.


```python
shape = (3, 4)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
a = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape) # a csr matrix as lhs
rhs = mx.nd.ones((4, 1))      # a dense vector as rhs
out = mx.nd.sparse.dot(a, rhs)  # invoke sparse dot operator specialized for dot(csr, dense)
{'out':out}
```

For any sparse operator, the storage type of output array is inferred based on inputs. You can either read the documentation or inspect the `stype` attribute of the output array to know what storage type is inferred:


```python
b = a * 2  # b will be a CSRNDArray since zero multiplied by 2 is still zero
c = a + mx.nd.ones(shape=(3, 4))  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```


For operators that don't specialize in sparse arrays, we can still use them with sparse inputs with some performance penalty. In MXNet, dense operators require all inputs and outputs to be in the dense format.

If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily, so that the dense operator can be used.

If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format.


```python
e = mx.nd.sparse.zeros('csr', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd.stype':d.stype, 'e.stype':e.stype} # stypes of a and e will be not changed
```


Note that warning messages will be printed when such a storage fallback event happens. If you are using jupyter notebook, the warning message will be printed in your terminal console.

### Data Loading

You can load data in batches from a CSRNDArray using `mx.io.NDArrayIter`:


```python
# Create the source CSRNDArray
data = mx.nd.array(np.arange(36).reshape((9,4))).tostype('csr')
labels = np.ones([9, 1])
batch_size = 3
dataiter = mx.io.NDArrayIter(data, labels, batch_size, last_batch_handle='discard')
# Inspect the data batches
[batch.data[0] for batch in dataiter]
```

You can also load data stored in the [libsvm file format](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) using `mx.io.LibSVMIter`, where the format is: ``<label> <col_idx1>:<value1> <col_idx2>:<value2> ... <col_idxN>:<valueN>``. Each line in the file records the label and the column indices and data for non-zero entries. For example, for a matrix with 6 columns, ``1 2:1.5 4:-3.5`` means the label is ``1``, the data is ``[[0, 0, 1,5, 0, -3.5, 0]]``. More detailed examples of `mx.io.LibSVMIter` are available in the [API documentation](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter).


```python
# Create a sample libsvm file in current working directory
import os
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data.t')
with open(data_path, 'w') as fout:
    fout.write('1.0 0:1 2:2\n')
    fout.write('1.0 0:3 5:4\n')
    fout.write('1.0 2:5 8:6 9:7\n')
    fout.write('1.0 3:8\n')
    fout.write('-1 0:0.5 9:1.5\n')
    fout.write('-2.0\n')
    fout.write('-3.0 0:-0.6 1:2.25 2:1.25\n')
    fout.write('-3.0 1:2 2:-1.25\n')
    fout.write('4 2:-1.2\n')

# Load CSRNDArrays from the file
data_train = mx.io.LibSVMIter(data_libsvm=data_path, data_shape=(10,), label_shape=(1,), batch_size=3)
for batch in data_train:
    print(data_train.getdata())
    print(data_train.getlabel())
```

Note that in the file the column indices are expected to be sorted in ascending order per row, and be zero-based instead of one-based.

### Advanced Topics

#### GPU Support

By default, `CSRNDArray` operators are executed on CPU. To create a `CSRNDArray` on a GPU, we need to explicitly specify the context:

**Note** If a GPU is not available, an error will be reported in the following section. In order to execute it a cpu, set `gpu_device` to `mx.cpu()`.


```python
import sys
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
try:
    a = mx.nd.sparse.zeros('csr', (100, 100), ctx=gpu_device)
    a
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```

## Train a Linear Regression Model with Sparse Symbols
In previous tutorials, we introduced `CSRNDArray` and `RowSparseNDArray`,
the basic data structures for manipulating sparse data.
MXNet also provides `Sparse Symbol` API, which enables symbolic expressions that handle sparse arrays.
In this tutorial, we first focus on how to compose a symbolic graph with sparse operators,
then train a linear regression model using sparse symbols with the Module API.

### Variables

Variables are placeholder for arrays. We can use them to hold sparse arrays too.

#### Variable Storage Types

The `stype` attribute of a variable is used to indicate the storage type of the array.
By default, the `stype` of a variable is "default" which indicates the default dense storage format.
We can specify the `stype` of a variable as "csr" or "row_sparse" to hold sparse arrays.


```python
import mxnet as mx
import numpy as np
import random

# set the seeds for repeatability
random.seed(42)
np.random.seed(42)
mx.random.seed(42)

# Create a variable to hold an NDArray
a = mx.sym.Variable('a')
# Create a variable to hold a CSRNDArray
b = mx.sym.Variable('b', stype='csr')
# Create a variable to hold a RowSparseNDArray
c = mx.sym.Variable('c', stype='row_sparse')
(a, b, c)
```



#### Bind with Sparse Arrays

The sparse symbols constructed above declare storage types of the arrays to hold.
To evaluate them, we need to feed the free variables with sparse data.

You can instantiate an executor from a sparse symbol by using the `simple_bind` method,
which allocate zeros to all free variables according to their storage types.
The executor provides `forward` method for evaluation and an attribute
`outputs` to get all the results. Later, we will show the use of the `backward` method and other methods computing the gradients and updating parameters. A simple example first:


```python
shape = (2,2)
# Instantiate an executor from sparse symbols
b_exec = b.simple_bind(ctx=mx.cpu(), b=shape)
c_exec = c.simple_bind(ctx=mx.cpu(), c=shape)
b_exec.forward()
c_exec.forward()
# Sparse arrays of zeros are bound to b and c
print(b_exec.outputs, c_exec.outputs)
```

You can update the array held by the variable by accessing executor's `arg_dict` and assigning new values.


```python
b_exec.arg_dict['b'][:] = mx.nd.ones(shape).tostype('csr')
b_exec.forward()
# The array `b` holds are updated to be ones
eval_b = b_exec.outputs[0]
{'eval_b': eval_b, 'eval_b.asnumpy()': eval_b.asnumpy()}
```

### Symbol Composition and Storage Type Inference

#### Basic Symbol Composition

The following example builds a simple element-wise addition expression with different storage types.
The sparse symbols are available in the `mx.sym.sparse` package.


```python
# Element-wise addition of variables with "default" stype
d = mx.sym.elemwise_add(a, a)
# Element-wise addition of variables with "csr" stype
e = mx.sym.sparse.negative(b)
# Element-wise addition of variables with "row_sparse" stype
f = mx.sym.sparse.elemwise_add(c, c)
{'d':d, 'e':e, 'f':f}
```

#### Storage Type Inference

What will be the output storage types of sparse symbols? In MXNet, for any sparse symbol, the result storage types are inferred based on storage types of inputs.
You can read the [Sparse Symbol API](http://mxnet.io/versions/master/api/python/symbol/sparse.html) documentation to find what output storage types are. In the example below we will try out the storage types introduced in the Row Sparse and Compressed Sparse Row tutorials: `default` (dense), `csr`, and `row_sparse`.


```python
add_exec = mx.sym.Group([d, e, f]).simple_bind(ctx=mx.cpu(), a=shape, b=shape, c=shape)
add_exec.forward()
dense_add = add_exec.outputs[0]
# The output storage type of elemwise_add(csr, csr) will be inferred as "csr"
csr_add = add_exec.outputs[1]
# The output storage type of elemwise_add(row_sparse, row_sparse) will be inferred as "row_sparse"
rsp_add = add_exec.outputs[2]
{'dense_add.stype': dense_add.stype, 'csr_add.stype':csr_add.stype, 'rsp_add.stype': rsp_add.stype}
```

#### Storage Type Fallback

For operators that don't specialize in certain sparse arrays, you can still use them with sparse inputs with some performance penalty. In MXNet, dense operators require all inputs and outputs to be in the dense format. If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily so that the dense operator can be used. If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format. Warning messages will be printed when such a storage fallback event happens.


```python
# `log` operator doesn't support sparse inputs at all, but we can fallback on the dense implementation
csr_log = mx.sym.log(a)
# `elemwise_add` operator doesn't support adding csr with row_sparse, but we can fallback on the dense implementation
csr_rsp_add = mx.sym.elemwise_add(b, c)
fallback_exec = mx.sym.Group([csr_rsp_add, csr_log]).simple_bind(ctx=mx.cpu(), a=shape, b=shape, c=shape)
fallback_exec.forward()
fallback_add = fallback_exec.outputs[0]
fallback_log = fallback_exec.outputs[1]
{'fallback_add': fallback_add, 'fallback_log': fallback_log}
```



#### Inspecting Storage Types of the Symbol Graph

When the environment variable `MXNET_INFER_STORAGE_TYPE_VERBOSE_LOGGING` is set to `1`, MXNet will log the storage type information of
operators' inputs and outputs in the computation graph. For example, we can inspect the storage types of
a linear classification network with sparse operators. Uncomment the line below and inspect your console.:


```python
# Set logging level for executor
import mxnet as mx
import os
#os.environ['MXNET_INFER_STORAGE_TYPE_VERBOSE_LOGGING'] = "1"
# Data in csr format
data = mx.sym.var('data', stype='csr', shape=(32, 10000))
# Weight in row_sparse format
weight = mx.sym.var('weight', stype='row_sparse', shape=(10000, 2))
bias = mx.symbol.Variable("bias", shape=(2,))
dot = mx.symbol.sparse.dot(data, weight)
pred = mx.symbol.broadcast_add(dot, bias)
y = mx.symbol.Variable("label")
output = mx.symbol.SoftmaxOutput(data=pred, label=y, name="output")
executor = output.simple_bind(ctx=mx.cpu())
```

### Training with Module APIs

In the following section we'll walk through how one can implement **linear regression** using sparse symbols and sparse optimizers.

The function you will explore is: *y = x<sub>1</sub>  +  2x<sub>2</sub> + ... 100x<sub>100*, where *(x<sub>1</sub>,x<sub>2</sub>, ..., x<sub>100</sub>)* are input features and *y* is the corresponding label.

#### Preparing the Data

In MXNet, both [mx.io.LibSVMIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter)
and [mx.io.NDArrayIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.NDArrayIter)
support loading sparse data in CSR format. In this example, we'll use the `NDArrayIter`.

You may see some warnings from SciPy. You don't need to worry about those for this example.


```python
# Random training data
feature_dimension = 100
train_data = mx.test_utils.rand_ndarray((1000, feature_dimension), 'csr', 0.01)
target_weight = mx.nd.arange(1, feature_dimension + 1).reshape((feature_dimension, 1))
train_label = mx.nd.dot(train_data, target_weight)
batch_size = 1
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, last_batch_handle='discard', label_name='label')
```

#### Defining the Model

Below is an example of a linear regression model specifying the storage type of the variables.


```python
initializer = mx.initializer.Normal(sigma=0.01)
X = mx.sym.Variable('data', stype='csr')
Y = mx.symbol.Variable('label')
weight = mx.symbol.Variable('weight', stype='row_sparse', shape=(feature_dimension, 1), init=initializer)
bias = mx.symbol.Variable('bias', shape=(1, ))
pred = mx.sym.broadcast_add(mx.sym.sparse.dot(X, weight), bias)
lro = mx.sym.LinearRegressionOutput(data=pred, label=Y, name="lro")
```

The above network uses the following symbols:

1. `Variable X`: The placeholder for sparse data inputs. The `csr` stype indicates that the array to hold is in CSR format.

2. `Variable Y`: The placeholder for dense labels.

3. `Variable weight`: The placeholder for the weight to learn. The `stype` of weight is specified as `row_sparse` so that it is initialized as RowSparseNDArray,
   and the optimizer will perform sparse update rules on it. The `init` attribute specifies what initializer to use for this variable.

4. `Variable bias`: The placeholder for the bias to learn.

5. `sparse.dot`: The dot product operation of `X` and `weight`. The sparse implementation will be invoked to handle `csr` and `row_sparse` inputs.

6. `broadcast_add`: The broadcasting add operation to apply `bias`.

7. `LinearRegressionOutput`: The output layer which computes *l2* loss against its input and the labels provided to it.

#### Training the model

Once we have defined the model structure, the next step is to create a module and initialize the parameters and optimizer.


```python
# Create module
mod = mx.mod.Module(symbol=lro, data_names=['data'], label_names=['label'])
# Allocate memory by giving the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# Initialize parameters by random numbers
mod.init_params(initializer=initializer)
# Use SGD as the optimizer, which performs sparse update on "row_sparse" weight
sgd = mx.optimizer.SGD(learning_rate=0.05, rescale_grad=1.0/batch_size, momentum=0.9)
mod.init_optimizer(optimizer=sgd)
```

Finally, we train the parameters of the model to fit the training data by using the `forward`, `backward`, and `update` methods in Module.


```python
# Use mean square error as the metric
metric = mx.metric.create('MSE')
# Train 10 epochs
for epoch in range(10):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print('Epoch %d, Metric = %s' % (epoch, metric.get()))
assert metric.get()[1] < 1, "Achieved MSE (%f) is larger than expected (1.0)" % metric.get()[1]
```


#### Training the model with multiple machines or multiple devices

Distributed training with `row_sparse` weights and gradients are supported in MXNet, which significantly reduces communication cost for large models. To train a sparse model with multiple machines, you need to call `prepare` before `forward`, or `save_checkpoint`.
Please refer to the example in [mxnet/example/sparse/linear_classification](https://github.com/apache/incubator-mxnet/tree/master/example/sparse/linear_classification)
for more details.
