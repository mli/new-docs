Data
======



Overview
---------------

This document lists the data APIs in Gluon:

.. autosummary::
    :nosignatures:

    mxnet.gluon.data
    mxnet.gluon.data.vision


The `Gluon Data` API, defined in the `gluon.data` package, provides useful dataset loading
and processing tools, as well as common public datasets.

In the rest of this document, we list routines provided by the `gluon.data` package.

Data
----

.. currentmodule:: mxnet.gluon.data


.. autosummary::
    :nosignatures:

    Dataset
    ArrayDataset
    RecordFileDataset


.. autosummary::
    :nosignatures:

    Sampler
    SequentialSampler
    RandomSampler
    BatchSampler


.. autosummary::
    :nosignatures:

    DataLoader


Vision
-------

Vision Datasets
^^^^^^^^^^^^^^^^^^

.. currentmodule:: mxnet.gluon.data.vision.datasets


.. autosummary::
    :nosignatures:

    MNIST
    FashionMNIST
    CIFAR10
    CIFAR100
    ImageRecordDataset
    ImageFolderDataset


Vision Transforms
^^^^^^^^^^^^^^^^^

.. currentmodule:: mxnet.gluon.data.vision.transforms


.. autosummary::
    :nosignatures:

    Compose
    Cast
    ToTensor
    Normalize
    RandomResizedCrop
    CenterCrop
    Resize
    RandomFlipLeftRight
    RandomFlipTopBottom
    RandomBrightness
    RandomContrast
    RandomSaturation
    RandomHue
    RandomColorJitter
    RandomLighting
