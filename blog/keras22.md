
# Announcing Keras-MXNet v2.2



Contributors of [Keras-MXNet](https://github.com/awslabs/keras-apache-mxnet/) are pleased to announce the release of [v2.2.0](https://github.com/awslabs/keras-apache-mxnet/releases/tag/2.2.0) which brings a number of key improvements to the package. Most notably, the package has been updated to include the changes brought in by [Keras v2.2.0](https://github.com/keras-team/keras/releases/tag/2.2.0). Massive backend design updates and a simplification of the API are the key highlights here. You can now use the new and simple Model API from Keras and get the high-performance of MXNet at the same time.

Keras-MXNet further improves the coverage of Keras operators with an MXNet backend. Critical operators like depthwise_conv2D, separable_conv2D, and conv1D* *with* causal padding* are supported by the MXNet backend in this release. Using these operators, you can now use MobileNet and Xception models with high-performance Keras-MXNet. Other layer-wise changes have been introduced too, such as deprecating the Merge layer in favor of the Concatenate layer.
 
The MXNet backend continues to be highly scalable and performant (as described in these [benchmarks](https://github.com/awslabs/keras-apache-mxnet/tree/master/benchmark) for the previous release). As before, we mark RNN support as experimental here. We also fix several bugs in this release and now support *Custom Loss* and *Custom Callbacks.*

**Quick to install**

Trying out the Keras-MXNet takes only a minute. First, install *keras-mxnet*:

    *pip install keras-mxnet*

If you’re using GPUs, install MXNet with CUDA 9 support:

    *pip install mxnet-cu90*

If you’re using CPU-only, install MXNet with MKL:

    *pip install mxnet-mkl*

Then train your Keras models with the MXNet backend and witness the speed increase! The [Keras examples](https://github.com/awslabs/keras-apache-mxnet/tree/master/examples) work out-of-the-box. To test out training at scale with multi-GPU training, run the [CIFAR10 multi-GPU script](https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/cifar10_resnet_multi_gpu.py). Usage of this script is covered in [AWS blog post’s CNN tutorial](https://aws.amazon.com/blogs/machine-learning/apache-mxnet-incubating-adds-support-for-keras-2/). The script expects four GPUs, but can be updated with the number of GPUs you’re running.

If you’re a Keras user, and you like where this is going, join the project, provide feedback, or pitch in on a feature you want to see. As an open source project, these great features are free to use, and are influenced and improved by open source community’s involvement. There are [calls for contribution](https://github.com/awslabs/keras-apache-mxnet/labels/help%20wanted) to enhance RNN support, which is [currently experimental](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/using_rnn_with_mxnet_backend.md). Also, make sure you follow Apache MXNet to keep posted on new features, like details on how you can use MXNet Model Server to serve your Keras-MXNet models!

* [GitHub](https://github.com/awslabs/keras-apache-mxnet)

* [Keras-MXNet v2.2.0 release notes](https://github.com/awslabs/keras-apache-mxnet/releases/tag/2.2.0)

* [Keras-MXNet docs](https://github.com/awslabs/keras-apache-mxnet/tree/master/docs/mxnet_backend)

* [Keras-MXNet benchmarks](https://github.com/awslabs/keras-apache-mxnet/tree/master/benchmark)

* [Train using Keras-MXNet and inference using MXNet Scala API](https://medium.com/apache-mxnet/train-using-keras-mxnet-and-inference-using-mxnet-scala-api-49476a16a46a)

Thanks to [Lai We](https://medium.com/@royweilai)i, [Kalyanee Chendke](https://medium.com/@kchendke), and [Thom Lane](https://medium.com/@thom.e.lane).
