``model_zoo``
==================

.. currentmodule:: mxnet.gluon.model_zoo

## Overview

This document lists the model APIs in Gluon:

.. autosummary::
    :nosignatures:

    mxnet.gluon.model_zoo
    mxnet.gluon.model_zoo.vision

The `Gluon Model Zoo` API, defined in the `gluon.model_zoo` package, provides pre-defined
and pre-trained models to help bootstrap machine learning applications.

In the rest of this document, we list routines provided by the `gluon.model_zoo` package.

### Vision

.. currentmodule:: mxnet.gluon.model_zoo.vision
.. automodule:: mxnet.gluon.model_zoo.vision

.. autosummary::
    :nosignatures:

    get_model

#### ResNet

.. autosummary::
    :nosignatures:

    resnet18_v1
    resnet34_v1
    resnet50_v1
    resnet101_v1
    resnet152_v1
    resnet18_v2
    resnet34_v2
    resnet50_v2
    resnet101_v2
    resnet152_v2

.. autosummary::
    :nosignatures:

    ResNetV1
    ResNetV2
    BasicBlockV1
    BasicBlockV2
    BottleneckV1
    BottleneckV2
    get_resnet

#### VGG

.. autosummary::
    :nosignatures:

    vgg11
    vgg13
    vgg16
    vgg19
    vgg11_bn
    vgg13_bn
    vgg16_bn
    vgg19_bn

.. autosummary::
    :nosignatures:

    VGG
    get_vgg

#### Alexnet

val_rst
.. autosummary::
    :nosignatures:

    alexnet



.. autosummary::
    :nosignatures:

    AlexNet


#### DenseNet


.. autosummary::
    :nosignatures:

    densenet121
    densenet161
    densenet169
    densenet201



.. autosummary::
    :nosignatures:

    DenseNet


#### SqueezeNet


.. autosummary::
    :nosignatures:

    squeezenet1_0
    squeezenet1_1



.. autosummary::
    :nosignatures:

    SqueezeNet


#### Inception


.. autosummary::
    :nosignatures:

    inception_v3



.. autosummary::
    :nosignatures:

    Inception3


#### MobileNet


.. autosummary::
    :nosignatures:

    mobilenet1_0
    mobilenet0_75
    mobilenet0_5
    mobilenet0_25
    mobilenet_v2_1_0
    mobilenet_v2_0_75
    mobilenet_v2_0_5
    mobilenet_v2_0_25



.. autosummary::
    :nosignatures:

    MobileNet
    MobileNetV2
