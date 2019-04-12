# Data

The `data` module provides APIs to 1) load and parse datasets, 2)
transform data examples, and 3) sample mini-batches for the training program.
This tutorial will go through these three functionalities. Let's first import
the modules needed, the key module is `mxnet.gluon.data`, which is imported as
`gdata` to avoid the too commonly used name `data`.

```{.python .input  n=1}
import numpy as np
from matplotlib import pyplot as plt
import tarfile
import time

from mxnet import nd, image, io
from mxnet.gluon import utils, data as gdata
```

## Load and Parse Datasets

To use a dataset, we first need to parse it into
individual examples with the `NDArray` data type. The base class is `Dataset`,
which defines two essential methods: `__getitem__` to access the i-th example
and `__len__` to get the number of examples. 

`ArrayDataset` is an
implementation of `Dataset` that combines multiple array-like objects into a
dataset. In the following example, we define NDArray features `X` with NDArray
label `y`, and then create the dataset.

```{.python .input  n=15}
features = nd.random.uniform(shape=(10, 3))
labels = nd.arange(10)
dataset = gdata.ArrayDataset(features, labels)
```

We can query the number examples in this dataset:

```{.python .input}
len(dataset)
```

And access an arbitrary example with its index. The returned example is a list
contains its features and label.

```{.python .input  n=11}
sample = dataset[1]
'features:', sample[0], 'label:', sample[1]
```

Note that the label for each example is a scalar, it is automatically converted
into numpy to make using as an index easy, e.g. we don't need to call
`sample[1].asscalar()`.

```{.python .input}
type(sample[1])
```

In addition, `ArrayDataset` can construct a dataset with any array-like objects,
with an arbitrary number of arrays:

```{.python .input  n=14}
dataset2 = gdata.ArrayDataset(features, np.random.uniform(size=(10,1)), list(range(0,10)))
sample = dataset2[1]
type(sample[0]), type(sample[1]), type(sample[2])
```

### Predefined Datasets

This module provides several commonly used datasets
that will be automatically downloaded during creation. For example, we can
obtain both the training and validation set of MNIST:

```{.python .input  n=2}
mnist_train = gdata.vision.MNIST()
mnist_valid = gdata.vision.MNIST(train=False)
print('# of training examples =', len(mnist_train))
print('# of validation examples =', len(mnist_valid))
```

Obtaining one example is as similar as before:

```{.python .input}
sample = mnist_train[1]
print('X shape:', sample[0].shape)
print('y:', sample[1])
```

Besides MNIST, `mxnet.gluon.data.vision` provides these three datasets:
FashionMNIST, CIFAR10,
and CIFAR100.

<!-- TODO: RecordFileDataset,
ImageRecordDataset -->

### Load Individual Images

In vision tasks, the
examples are often stored as individual images files. If we place images within
each category in separate folders, then we can use `ImageFolderDataset` to load
both images and labels. 

Let's download a tiny image dataset as an example.

```{.python .input}
utils.download('https://github.com/dmlc/web-data/raw/master/mxnet/doc/dogcat.tar.gz')
with tarfile.open('dogcat.tar.gz') as f:
    f.extractall()
```

Then check contents within this dataset:

```{.python .input}
# You may need to install the `tree' program, such as uncommenting the following line for Ubuntu:
# !sudo apt-get install tree
!tree dogcat    
```

As can be seen, it has two categories, dog and cat. Image files are placed in
subfolders with categories as folder names. Now construct an
`ImageFolderDataset` instance with specifying the dataset root folder.

```{.python .input}
dogcat = gdata.vision.ImageFolderDataset('./dogcat')
```

We can access all categories through the attribute `synsets`:

```{.python .input}
dogcat.synsets
```

Next let's print a particular sample with its label:

```{.python .input}
sample = dogcat[1]
plt.imshow(sample[0].asnumpy())
plt.show()
'label:', dogcat.synsets[sample[1]]
```

## Transform Data Examples

The raw data examples often need to be transformed
before feeding into a neural network. Class `Dataset` provides two methods
`transform` and `transform_first` to allow users to specify the transformation
methods.

In the following example, we define a function to resize an image into
200px height and 300px width. And then we pass it into the dataset through the
`transform_first` method, which returns a new dataset with the transformations
recorded.

```{.python .input}
def resize(x):
    y = image.imresize(x, 300, 200)
    print('resize', x.shape, 'into', y.shape)
    return y
dogcat_resized = dogcat.transform_first(resize)
```

As can be seen, transformations are applied when accessing examples, which is
necessary when these transformations contain randomness. But we can also apply
all transformations during creating the dataset.

```{.python .input}
dogcat_cached = dogcat.transform_first(resize, lazy=False)
```

So no transform is needed during accessing examples.

```{.python .input}
dogcat_cached[0][0].shape
```

Besides `transform_first`, we can apply transform to all entries in an example.
The following examples add the label by 10 besides resizing an image:

```{.python .input}
dogcat_both = dogcat.transform(lambda x, y: (resize(x), y+10))
dogcat_both[0][1]
```

The `vision.transform` submodule provide multiple pre-defined data
transformation methods. For example, the following example chains the image
resize and `ToTensor`, which transforms the data layout into (C x H x W) with
float32 data type. Please refer to [Image Augmentation](./image-augmentation.md)
for more details.

```{.python .input  n=3}
transforms = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize((24, 24)),
    gdata.vision.transforms.ToTensor()])
mnist_transformed = mnist_train.transform_first(transforms, lazy=True)
(mnist_train[0][0].shape, '->', mnist_transformed[0][0].shape)
```

## Sample Mini-batches 

If we train a neural network with mini-batch SGD, we
need to sample a mini-batch for every iteration. Class `DataLoader` samples a
dataset into mini-batches. In the following example, we create a `DataLoader`
instance, which is an iterator that returns a mini-batch each time.

```{.python .input  n=18}
data = gdata.DataLoader(dataset, batch_size=4)
for X, y in data:
    print('X shape:', X.shape, '\ty:', y.asnumpy())
```

Since the number of examples can not be divided by the batch size, the last min-
batch only has two examples. We can chose to ignore the last incomplete mini-
batch:

```{.python .input  n=19}
data = gdata.DataLoader(dataset, batch_size=4, last_batch='discard')
for X, y in data:
    print('y:', y.asnumpy())
```

Or put it into the beginning of the next epoch:

```{.python .input  n=23}
data = gdata.DataLoader(dataset, batch_size=4, last_batch='rollover')
for X, y in data:
    print('epoch 0, y:', y.asnumpy())
for X, y in data:
    print('epoch 1, y:', y.asnumpy())
```

In mini-batch SGD, a mini-batch needs to consist of randomly sampled examples.
We can set the `shuffle` argument to get random batches:

```{.python .input}
data = gdata.DataLoader(dataset, batch_size=4, shuffle=True)
for X, y in data:
    print('y:', y.asnumpy())
```

### Customize Sampling

`DataLoader` reads examples either sequentially or
uniformly randomly without
replacement. We can change this behavior through
customized samplers. An sampler
is an iterator returning an sample index each
time. For example, we create an
sampler that first sequentially reads even
indexes and then odd indexes.

```{.python .input  n=30}
class MySampler():
    def __init__(self, length):
        self.len = length
    def __iter__(self):
        for i in list(range(0,self.len,2))+list(range(1,self.len,2)):
            yield i
data = gdata.DataLoader(dataset, batch_size=4, sampler=MySampler(len(dataset)))
for X, y in data:
    print(y.asnumpy())
```

Similarly, we can change how a mini-batches is sampled through the
`batch_sampler` argument. 

### Multi-process

Reading data is often one of the
major performance bottlenecks. We can accelerate it through multi-process (only
Linux and Macos are supported.) Let's first benchmark the time to read the MNIST
training set:

```{.python .input  n=7}
tic = time.time()
data = gdata.DataLoader(mnist_transformed, batch_size=64)
for X, y in data:
    pass
'%.1f sec' % (time.time() - tic)
```

Now let's use 4 processes:

```{.python .input  n=8}
tic = time.time()
data = gdata.DataLoader(mnist_transformed, batch_size=64, num_workers=4)
for X, y in data:
    pass
'%.1f sec' % (time.time() - tic)
```

## Appendix: From `DataIter` to `DataLoader`

Before Gluon's `DataLoader`, MXNet
provides `DataIter` in the `io` module to read mini-batches. They are similar to
each other but `DataLoader` often returns a tuple of `(feature, label)` for a
mini-batches, while `DataIter` returns a `DataBatch`. The following example
wraps a `DataIter` into a `DataLoader` so you can reuse the existing codes but
enjoys the benefits of Gluon.

```{.python .input}
class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter
    def __iter__(self):
        self.data_iter.reset()
        return self
    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label
    def next(self):
        return self.__next__() # for Python 2
```

Now create an `DataIter` instance, and then get the according `DataLoader`
wrapper.

```{.python .input}
data_iter = io.NDArrayIter(data=features, label=labels, batch_size=4)
data = DataIterLoader(data_iter)
for X, y in data:
    print('X shape:', X.shape, '\ty:', y.asnumpy())
```
