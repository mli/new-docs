# An Intro: Manipulate Data the MXNet Way with NDArray

## Overview
This guide will introduce you to how data is handled with MXNet. You will learn the basics about MXNet's multi-dimensional array format, `ndarray`.

The content was extracted and simplified from the gluon tutorials in [The Straight Dope](https://gluon.mxnet.io/).

## Prerequisites
* [MXNet installed in a Python environment](https://mxnet.incubator.apache.org/install/index.html?language=Python).
* Python 2.7.x or Python 3.x


## Operations

NDArray supports a large number of standard mathematical operations. Such as element-wise addition:
<!-- keeping it easy -->


```{.python .input}
print('x=', x)
print('y=', y)
x = x + y
print('x = x + y, x=', x)
```

    x=
    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]
    <NDArray 3x4 @cpu(0)>
    y=
    [[ 0.03629481 -0.67765152 -0.49024421  0.10073948]
     [-0.95017916  0.57595438  0.03751944 -0.3469252 ]
     [-0.72984636 -0.22134334 -2.04010558 -1.80471897]]
    <NDArray 3x4 @cpu(0)>
    x = x + y, x=
    [[ 1.03629482  0.32234848  0.50975579  1.10073948]
     [ 0.04982084  1.57595444  1.03751945  0.6530748 ]
     [ 0.27015364  0.77865666 -1.04010558 -0.80471897]]
    <NDArray 3x4 @cpu(0)>


Multiplication:


```{.python .input}
x = nd.array([1, 2, 3])
y = nd.array([2, 2, 2])
x * y
```





    [ 2.  4.  6.]
    <NDArray 3 @cpu(0)>



And exponentiation:
<!-- with these next ones we'll just have to take your word for it... -->


```{.python .input}
nd.exp(x)
```





    [  2.71828175   7.38905621  20.08553696]
    <NDArray 3 @cpu(0)>



We can also grab a matrix's transpose to compute a proper matrix-matrix product.
<!-- because we need to do that before we have coffee every day... and you know how those dirty, improper matrixeses can be... -->


```{.python .input}
nd.dot(x, y.T)
```





    [ 12.]
    <NDArray 1 @cpu(0)>



We'll explain these operations and present even more operators in the [linear algebra](P01-C03-linear-algebra.ipynb) chapter. But for now, we'll stick with the mechanics of working with NDArrays.

## In-place operations

In the previous example, every time we ran an operation, we allocated new memory to host its results. For example, if we write `y = x + y`, we will dereference the matrix that `y` used to point to and instead point it at the newly allocated memory. We can show this using Python's `id()` function, which tells us precisely which object a variable refers to.

<!-- dereference is something C++ people would know but everyone else... not so much. What's the point? ;) get it? Put it in more context as to why you care about this and why this is in front of so much other material. Seems like an optimization topic best suited for later...
###edit### we just talked about this, so I have better context. Now I understand, but your new reader will not. This should be covered in much more detail, and quite possibily in its own notebook since I think it will help to show some gotchas like you mentioned verbally. I am still leaning toward delaying the introduction of this topic....-->


```{.python .input}
print('y=', y)
print('id(y):', id(y))
y = y + x
print('after y=y+x, y=', y)
print('id(y):', id(y))
```

    y=
    [ 2.  2.  2.]
    <NDArray 3 @cpu(0)>
    id(y): 4630663856
    after y=y+x, y=
    [ 3.  4.  5.]
    <NDArray 3 @cpu(0)>
    id(y): 4630567792


We can assign the result to a previously allocated array with slice notation, e.g., `result[:] = ...`.


```{.python .input}
print('x=', x)
z = nd.zeros_like(x)
print('z is zeros_like x, z=', z)
print('id(z):', id(z))
print('y=', y)
z[:] = x + y
print('z[:] = x + y, z=', z)
print('id(z) is the same as before:', id(z))
```

    x=
    [ 1.  2.  3.]
    <NDArray 3 @cpu(0)>
    z is zeros_like x, z=
    [ 0.  0.  0.]
    <NDArray 3 @cpu(0)>
    id(z): 4630567232
    y=
    [ 3.  4.  5.]
    <NDArray 3 @cpu(0)>
    z[:] = x + y, z=
    [ 4.  6.  8.]
    <NDArray 3 @cpu(0)>
    id(z) is the same as before: 4630567232


However, `x+y` here will still allocate a temporary buffer to store the result before copying it to z. To make better use of memory, we can perform operations in place, avoiding temporary buffers. To do this we specify the `out` keyword argument every operator supports:


```{.python .input}
print('x=', x, 'is in id(x):', id(x))
print('y=', y, 'is in id(y):', id(y))
print('z=', z, 'is in id(z):', id(z))
nd.elemwise_add(x, y, out=z)
print('after nd.elemwise_add(x, y, out=z), x=', x, 'is in id(x):', id(x))
print('after nd.elemwise_add(x, y, out=z), y=', y, 'is in id(y):', id(y))
print('after nd.elemwise_add(x, y, out=z), z=', z, 'is in id(z):', id(z))
```

    x=
    [ 1.  2.  3.]
    <NDArray 3 @cpu(0)> is in id(x): 4630588048
    y=
    [ 3.  4.  5.]
    <NDArray 3 @cpu(0)> is in id(y): 4630567792
    z=
    [ 4.  6.  8.]
    <NDArray 3 @cpu(0)> is in id(z): 4630567232
    after nd.elemwise_add(x, y, out=z), x=
    [ 1.  2.  3.]
    <NDArray 3 @cpu(0)> is in id(x): 4630588048
    after nd.elemwise_add(x, y, out=z), y=
    [ 3.  4.  5.]
    <NDArray 3 @cpu(0)> is in id(y): 4630567792
    after nd.elemwise_add(x, y, out=z), z=
    [ 4.  6.  8.]
    <NDArray 3 @cpu(0)> is in id(z): 4630567232


If we're not planning to re-use ``x``, then we can assign the result to ``x`` itself. There are two ways to do this in MXNet.
1. By using slice notation x[:] = x op y
2. By using the op-equals operators like `+=`


```{.python .input}
print('x=', x, 'is in id(x):', id(x))
x += y
print('x=', x, 'is in id(x):', id(x))
```

    x=
    [ 1.  2.  3.]
    <NDArray 3 @cpu(0)> is in id(x): 4630588048
    x=
    [ 4.  6.  8.]
    <NDArray 3 @cpu(0)> is in id(x): 4630588048


## Slicing
MXNet NDArrays support slicing in all the ridiculous ways you might imagine accessing your data. For a quick review:

```
a[start:end] # items start through end-1
a[start:]    # items start through the rest of the array
a[:end]      # items from the beginning through end-1
a[:]         # a copy of the whole array
```

Here's an example of reading the second and third rows from `x`.


```{.python .input}
x = nd.array([1, 2, 3])
print('1D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
x = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('multi-D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
```

    1D complete array, x=
    [ 1.  2.  3.]
    <NDArray 3 @cpu(0)>
    slicing the 2nd and 3rd elements, s=
    [ 2.  3.]
    <NDArray 2 @cpu(0)>
    multi-D complete array, x=
    [[  1.   2.   3.   4.]
     [  5.   6.   7.   8.]
     [  9.  10.  11.  12.]]
    <NDArray 3x4 @cpu(0)>
    slicing the 2nd and 3rd elements, s=
    [[  5.   6.   7.   8.]
     [  9.  10.  11.  12.]]
    <NDArray 2x4 @cpu(0)>


Now let's try writing to a specific element.


```{.python .input}
print('original x, x=', x)
x[2] = 9.0
print('replaced entire row with x[2] = 9.0, x=', x)
x[0,2] = 9.0
print('replaced specific element with x[0,2] = 9.0, x=', x)
x[1:2,1:3] = 5.0
print('replaced range of elements with x[1:2,1:3] = 5.0, x=', x)
```

    original x, x=
    [[ 1.  2.  9.  4.]
     [ 5.  5.  5.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>
    replaced entire row with x[2] = 9.0, x=
    [[ 1.  2.  9.  4.]
     [ 5.  5.  5.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>
    replaced specific element with x[0,2] = 9.0, x=
    [[ 1.  2.  9.  4.]
     [ 5.  5.  5.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>
    replaced range of elements with x[1:2,1:3] = 5.0, x=
    [[ 1.  2.  9.  4.]
     [ 5.  5.  5.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>


Multi-dimensional slicing is also supported.


```{.python .input}
x = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('original x, x=', x)
s = x[1:2,1:3]
print('plucking specific elements with x[1:2,1:3]', s)
s = x[:,:1]
print('first column with x[:,:1]', s)
s = x[:1,:]
print('first row with x[:1,:]', s)
s = x[:,3:]
print('last column with x[:,3:]', s)
s = x[2:,:]
print('last row with x[2:,:]', s)
```

    original x, x=
    [[  1.   2.   3.   4.]
     [  5.   6.   7.   8.]
     [  9.  10.  11.  12.]]
    <NDArray 3x4 @cpu(0)>
    plucking specific elements with x[1:2,1:3]
    [[ 6.  7.]]
    <NDArray 1x2 @cpu(0)>
    first column with x[:,:1]
    [[ 1.]
     [ 5.]
     [ 9.]]
    <NDArray 3x1 @cpu(0)>
    first row with x[:1,:]
    [[ 1.  2.  3.  4.]]
    <NDArray 1x4 @cpu(0)>
    last column with x[:,3:]
    [[  4.]
     [  8.]
     [ 12.]]
    <NDArray 3x1 @cpu(0)>
    last row with x[2:,:]
    [[  9.  10.  11.  12.]]
    <NDArray 1x4 @cpu(0)>


## Broadcasting

You might wonder, what happens if you add a vector `y` to a matrix `X`? These operations, where we compose a low dimensional array `y` with a high-dimensional array `X` invoke a functionality called broadcasting. First we'll introduce `.arange` which is useful for filling out an array with evenly spaced data. Then we can take the low-dimensional array and duplicate it along any axis with dimension $1$ to match the shape of the high dimensional array.
Consider the following example.

Comment (visible to demonstrate with font): dimension one(1)? Or L(elle) or l(lil elle) or I(eye) or... ? We don't even use the notation later, so did it need to be introduced here?

<!--Also, if you use a shape like (3,3) you lose some of the impact and miss some errors if people play with the values. Better to have a distinct shape so that it is more obvious what is happening and what can break.-->


```{.python .input}
x = nd.ones(shape=(3,6))
print('x = ', x)
y = nd.arange(6)
print('y = ', y)
print('x + y = ', x + y)
```

    x =
    [[ 1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.]]
    <NDArray 3x6 @cpu(0)>
    y =
    [ 0.  1.  2.  3.  4.  5.]
    <NDArray 6 @cpu(0)>
    x + y =
    [[ 1.  2.  3.  4.  5.  6.]
     [ 1.  2.  3.  4.  5.  6.]
     [ 1.  2.  3.  4.  5.  6.]]
    <NDArray 3x6 @cpu(0)>


While `y` is initially of shape (6),
MXNet infers its shape to be (1,6),
and then broadcasts along the rows to form a (3,6) matrix).
You might wonder, why did MXNet choose to interpret `y` as a (1,6) matrix and not (6,1).
That's because broadcasting prefers to duplicate along the left most axis.
We can alter this behavior by explicitly giving `y` a 2D shape using `.reshape`. You can also chain `.arange` and `.reshape` to do this in one step.


```{.python .input}
y = y.reshape((3,1))
print('y = ', y)
print('x + y = ', x+y)
y = nd.arange(6).reshape((3,1))
print('y = ', y)
```

    y =
    [[ 0.]
     [ 1.]
     [ 2.]]
    <NDArray 3x1 @cpu(0)>
    x + y =
    [[ 1.  1.  1.  1.  1.  1.]
     [ 2.  2.  2.  2.  2.  2.]
     [ 3.  3.  3.  3.  3.  3.]]
    <NDArray 3x6 @cpu(0)>
    y =
    [[ 0.]
     [ 1.]
     [ 2.]]
    <NDArray 3x1 @cpu(0)>


## Converting from MXNet NDArray to NumPy
Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do not share memory.


```{.python .input}
a = x.asnumpy()
type(a)
```




    numpy.ndarray




```{.python .input}
y = nd.array(a)
print('id(a)=', id(a), 'id(x)=', id(x), 'id(y)=', id(y))
```

    id(a)= 4630961328 id(x)= 4631751144 id(y)= 4631748504


## Next Up

[NDArray Contexts](ndarray-contexts.md)
