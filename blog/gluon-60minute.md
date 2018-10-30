
# MXNet Gluon in 60-minutes

Gluon is an imperative API for MXNet that’s flexible and easy-to-use, and with our new 60-Minute Crash Course you can get up and running with Gluon straight away. You’ll learn the core concepts required to train neural networks with Gluon, including NDArray and Autograd. And we cover more advanced topics such as using Multiple GPUs to unlock the full potential of MXNet. We’re sure it will be 60 minutes well spent!

![](https://cdn-images-1.medium.com/max/3888/1*1T7xFfYGsiEDfi-sBDztDw.png)

We provide a dedicated [website](https://gluon-crash-course.mxnet.io/index.html) for the course and have a supplementary [YouTube video playlist](https://www.youtube.com/playlist?list=PLkEvNnRk8uVmVKRDgznk3o3LxmjFRaW7s) that walks you through the series. All chapters are available for download as Jupyter Notebooks, so you can try everything out while you follow along. And if you’ve got any questions along the way don’t be shy about posting on the [discussion forum](https://discuss.mxnet.io/).

### Chapter 1: Setup & NDArray

You just need MXNet installed to get started. On most platforms it’s as simple as pip install mxnet but you can find more detailed instructions [here](https://mxnet.incubator.apache.org/install/index.html). We use GPUs in the last chapter so [AWS SageMaker](https://aws.amazon.com/sagemaker/) is a great way to get setup quickly; GPU instances already have CUDA and notebook support right out-of-the-box. Also check out the [DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) if you’re familiar with AWS EC2.

In [this chapter](https://gluon-crash-course.mxnet.io/ndarray.html), we discuss the benefits of the Gluon API (when compared with Module API) for MXNet, and start with an introduction to NDArray: a fundamental concept when working with neural networks.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries" frameborder="0" allowfullscreen></iframe></center>

### Chapter 2: Defining Neural Networks

We create our first neural network in [this chapter](https://gluon-crash-course.mxnet.io/nn.html), starting with a single fully connected layer and working up to a custom network architecture. Along the way we implement the LeNet convolutional network, and discuss initialization of network parameters.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries" frameborder="0" allowfullscreen></iframe></center>

### Chapter 3: Automatic differentiation

Automatic differentiation is an incredibly useful feature of MXNet Gluon, as it handles gradient calculations for you when designing neural networks of all complexities. We take a look at the autograd package of MXNet Gluon in [this chapter](https://gluon-crash-course.mxnet.io/autograd.html), and get started with some simple examples.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries" frameborder="0" allowfullscreen></iframe></center>

### Chapter 4: Training Neural Networks

In [this chapter](https://gluon-crash-course.mxnet.io/train.html), we train a clothing classifier using MXNet Gluon and the FashionMNIST dataset. We take a look at our first training loop, and optimize our model using Stochastic Gradient Descent (SGD). Saving our model at the end, we reuse this model in the next few chapters.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries" frameborder="0" allowfullscreen></iframe></center>

### Chapter 5: Using Pre-trained Networks

In [this chapter](https://gluon-crash-course.mxnet.io/predict.html), we test the model from the previous video with unseen images of clothes from the FashionMNIST dataset. We then take a look at the Gluon Model Zoo and use a ResNet 50 model that’s been pre-trained on ImageNet to classify dog breeds.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries" frameborder="0" allowfullscreen></iframe></center>

### Chapter 6: Using GPUs

In our [last chapter](https://gluon-crash-course.mxnet.io/use_gpus.html) of the series, we take a look at using GPU to speed up training and inference of neural networks. We recommend using [AWS SageMaker](https://aws.amazon.com/sagemaker/) with GPU instances if you don’t have your own GPU because you get CUDA and notebook support right out-of-the-box. After running through the basics, we move to Multi-GPU training to use the full capabilities of MXNet Gluon.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries" frameborder="0" allowfullscreen></iframe></center>

### Graduated from the Crash Course?

We hope you enjoyed it and learnt a lot. After finishing the course you can continue learning from a wide range of [tutorials](https://mxnet.incubator.apache.org/tutorials/index.html) and [examples](https://github.com/apache/incubator-mxnet/tree/master/example) found on the MXNet website.
