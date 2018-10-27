
# A Way to Benchmark Your Deep Learning Framework On-premise

MXNet Makes Us Faster And Stronger!

### Why MXNet and Gluon?

Recently, our organization(@Machine Learning Cell, SK Telecom) started actively using MXNet and Gluon. As part of the DevOps organization, we need not only to develop new models constantly but also to maintain and deploy them seamlessly, even in a disruptive environment.

There are many deep learning frameworks these days: MXNet, TensorFlow, PyTorch, Theano, to name a few. But, from the “Ops” point of view, maintaining many models for all different frameworks can be challenging. Therefore, we need to choose a few deep learning frameworks by considering the following criteria:

* Convenience of code design

* High-quality reference code samples

* Speed of training and inference

* How many functions/methods are offered out of the box

* Version update compatibility

* Supporting multiple programming languages

TensorFlow has many high-quality reference code and tutorials. However, it is hard to debug because the imperative environment (TensorFlow’s eager execution) like PyTorch is still experimental. In contrast, PyTorch is good for research, but presents some issues in maintainability due to the frequent API breaking changes.

Fortunately, MXNet offers both low level-control and high-level APIs such as Gluon. MXNet and Gluon enable us to quickly test, operate and manage experimental models. In particular, Gluon offers more functions via gluon-cv (Computer Vision) and gluon-nlp (Natural Language Processing). The MXNet community is committed to respect semantic versioning, which means that no API breaking change are introduced between minor version updates of MXNet.

![Deep Learning Frameworks (source: [https://aws.amazon.com/ko/machine-learning/amis/](https://aws.amazon.com/ko/machine-learning/amis/))](https://cdn-images-1.medium.com/max/2048/1*7C46N8tlMQg28Y631HUZDQ.jpeg)*Deep Learning Frameworks (source: [https://aws.amazon.com/ko/machine-learning/amis/](https://aws.amazon.com/ko/machine-learning/amis/))*

## How to evaluate a deep learning framework?

**The Importance of Benchmark**
> Scenario 1. How do I know it is a good choice to migrate from Tensorflow to MXNet?

It is often difficult to compare results between frameworks because model performance metrics such as accuracy or speed, are affected not only by the framework but also by many other parameters. As a result, [**Keras-MXNet](https://github.com/awslabs/keras-apache-mxnet)** is a good choice for standardizing comparisons because we can test the same model definition with different frameworks, simply by changing the keras backend. We can install keras-mxnet easily using pip.
> Scenario 2. How do I know if it is a good choice to update from MXNet 1.1.0 to MXNet 1.2.0?

When it comes to comparing different versions of frameworks, it is relatively easy to perform benchmark tests if the framework is backwards compatible and if the benchmark code is already prepared. The most important point in this case to get well-defined benchmark test metrics.

**Benchmark Configuration**

How do you configure the benchmark on-premise? We recommend to list all the different environments, frameworks, models and metrics that are relevant to your use-case(s). For example:

    (1) Systems: CPUs, 1 GPU, 4 GPUs

    (2) Frameworks: TensorFlow, MXNet

    (3) Models: ResNet50, lstm

    (4) Metrics: Training Time, Training Accuracy

**1. Systems**

* A docker image with clear requirements makes it easy to setup a reproducible environment, including the number of GPUs to be used for the test.

* You can use the docker image ([link](https://github.com/keras-team/keras/tree/master/docker)) provided by keras with anaconda. But for my benchmark test, I created my own ([link](https://github.com/soeque1/benchmark_keras-mxnet/tree/master/docker)) since our system on-premise employs native python rather than anaconda.

* (GPU) It is easy to enable GPU with docker using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)[.](https://github.com/NVIDIA/nvidia-docker.)

* (CPU) If you would like to install MXNet with the DNN optimized Intel MKL-DNN library, you only have to run the following pip command:

    pip install mxnet-mkl

* Often times, the analytics server with your training data is blocked from accessing the internet for various security reasons. To overcome this, you can build the docker image from a server with internet access, save the image, and move it to the analytics server to run your tests with. For example:

    (in the server with internet)
    docker build -t bench_gpu:0.0.1 -f bench_gpu/Dockerfile ./
    docker save bench_gpu:0.0.1 > ./bench_gpu.tar

    (in the analyst server without internet)
    docker load < ./bench_gpu.tar

**2. Frameworks**

* You can run the same benchmark code for both of TensorFlow and MXNet easily with keras-mxnet. You may need to modify benchmark code if you want to compare the Gluon model though. Here is an example benchmark [(link)](https://gist.github.com/soeque1/d7c2f3406f66ea5211d9030f54a1b472).

**3. and 4. Models and Metrics**

* This model in keras-mxnet can be used as benchmark ([link](https://github.com/awslabs/keras-apache-mxnet/tree/master/benchmark)).

* The accuracy can be affected by hyper-parameters or initial weights, so only the training speed was measured.

* The metrics provided by the keras-mxnet package can be used directly. To also track inference time and accuracy metrics, you need to add them manually.

### Running the Benchmark

Here is an illustration of our benchmark setup:

![The Architecture of The Benchmark](https://cdn-images-1.medium.com/max/2000/1*Vzd9jsy5SZMzj8ug1zBCBg.png)*The Architecture of The Benchmark*

1. When the docker container is created, you specify the name of the docker image, the number of GPU, the chosen framework, storage location for results and so on.

    run_docker(system=gpu, framework=mxnet, storage=mystore)

2. Running the benchmark code in the docker container.

    run_benchmark(model=resnet50)

3. Storing the logs to the final location

    move_result(log)

<iframe src="https://medium.com/media/b3a33bdd01a81996b530987e744046f9" frameborder=0></iframe>

The above code snippet sequentially measure the performance of (1) the system (2) the framework, and (3) the model. The full code can be found [here](https://github.com/soeque1/benchmark_keras-mxnet/blob/master/run_docker_benchmarks.sh).

    for system in 4_gpu gpu cpu
      for model in resnet50
        for framework in mxnet_1.2.0 tensorflow_1.8.0
            run_docker(system, framework, storage, )
            run_benchmark(model, )
            move_result(log, )
> **By extending this scenario, you can use it for regression tests. It helps you find errors caused by Python and Python packages on specific models.**
> **Once you have stored benchmark results from various version of framework, you are able to monitor the Scenario 2 (issues across framework versions) at the same time.**

### The Benchmark Result

Here we present [the benchmark](https://github.com/soeque1/benchmark_keras-mxnet/blob/master/run_docker_benchmarks.sh) [results](https://github.com/soeque1/benchmark_keras-mxnet/blob/master/summary_benchmarks.ipynb) for MXNet against Tensorflow on our on-premise system. Just as shown in the [keras-mxnet benchmark test on the cloud](https://github.com/awslabs/keras-apache-mxnet/tree/master/benchmark), MXNet is faster than TensorFlow on ResNet50 in our environment.
> This benchmark was performed at NVIDIA’s DGX-1 Tesla V100

![Training time of ResNet50 between frameworks(5 repeats, 4-GPUs (left) 1-GPU(middle), CPUs(right))](https://cdn-images-1.medium.com/max/3654/1*9CSpVNtk9SyvRc-avhSNQg.png)*Training time of ResNet50 between frameworks(5 repeats, 4-GPUs (left) 1-GPU(middle), CPUs(right))*

## Extensions: Add your model

You may want to register your own model. This example shows how to add a simple MultiLayer Perceptron (MLP) model to the benchmark system we presented above. ([tensorflow/benchmark](https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/models/mnist_mlp_benchmark.py) was used as reference)

You can add a model by registering the python script that contains code for your model, in [model_config.py](https://github.com/awslabs/keras-apache-mxnet/blob/master/benchmark/scripts/models/model_config.py) as follows.

<iframe src="https://medium.com/media/7d73603b8381325a1e5835ac7925f569" frameborder=0></iframe>

The graph below shows MXNet is also faster than TensorFlow with the MLP model.

![Training time of the MLP between frameworks(5 repeats, 4-GPUs (left) 1-GPU(middle), CPUs(right))](https://cdn-images-1.medium.com/max/3648/1*qkfpjH8XY-y9aX0fPxaHhw.png)*Training time of the MLP between frameworks(5 repeats, 4-GPUs (left) 1-GPU(middle), CPUs(right))*

### Summary

* MXNet supports both imperative programming* *and symbolic programming*, *which makes it easier for developers to debug and manage the model. And it supports multiple languages such as Python, R, Scala, Julia, and C++.

* Setting aside the strong points mentioned above, according to our experiments, MXNet shows great performance in terms of training speed, in addition to a lot of features other frameworks are missing.

* Keras-mxnet can be a good choice for comparing quickly different deep learning frameworks performance.

* The benchmark helps you compare performance within-framework as well as between-frameworks.
