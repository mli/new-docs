
# How Curalate uses MXNet on AWS for some Deep Learning Magic

written by Louis K & Jesse Brizzi — July 27th, 2018

![](https://cdn-images-1.medium.com/max/2794/1*neCCKXr7CMh-ukhz2dGlCg.jpeg)

At [Curalate](https://www.curalate.com/), we use state of the art deep learning and computer vision to add a layer of magic to our products. [Intelligent Product Tagging](https://www.curalate.com/blog/intelligent-product-tagging/), for example, identifies our clients’ products in user-generated photos. Being a startup, we need to build these deep learning and computer vision systems the same way we do the rest of our products: quickly.

Our computer vision systems are built in [two phases](https://www.curalate.com/blog/how-Curalate-built-a-kick-ass-research-development-team/), research and productization, and we require a deep learning framework that accelerates both. During the research phase, we need a framework that’s quick to get started with and is flexible enough to experiment on with new ideas. Once we have a solution, we need a framework that can easily be integrated into a microservice and deployed to multiple production environments.

In the past, we used [Caffe](http://caffe.berkeleyvision.org/) for experimentation and our own custom inference interface to deploy the trained models to production. Experimentation was slow due to Caffe’s dated Python API, lack of automatic differentiation, unreliable build/install process, and clunky support for advanced layers which required us to maintain our own custom fork. Productization of Caffe was challenging since we had to maintain our own [JNI interface](http://engineering.curalate.com/2016/04/29/bridging-scala-to-c++-with-bridj.html).

We needed a new and modern framework that fulfilled all of our needs while saving us from the shortcomings of Caffe. After a [review of all the available options](http://engineering.curalate.com/2018/03/23/DL-lib-for-app-dev-and-prod.html), we decided to move to [MXNet](https://mxnet.incubator.apache.org/). In this post, we’ll discuss why we migrated to MXNet as our deep learning framework of choice to facilitate our speed of experimentation, development, and deployment.

## Training and Experimentation

Whenever we are faced with a new computer vision problem, we start by looking at existing state-of-the-art implementations. If we are lucky the functionality of the service we are implementing is similar to an existing pre-trained model for MXNet. MXNet has a fairly fleshed out and maintained [Model Zoo](https://mxnet.incubator.apache.org/model_zoo/) that contains all of the standard pre-trained models that we would expect from any deep learning framework (ImageNet, PascalVOC, …). If we can not find what we need there, MXNet is also one of the frameworks that [supports](https://aws.amazon.com/blogs/machine-learning/announcing-onnx-support-for-apache-mxnet/) [ONNX](https://onnx.ai/) and its cross framework model format giving us access to many more pre-trained models.

Converting and reusing old models and code is also possible with MXNet. [MMDNN](https://github.com/Microsoft/MMdnn) provides support for converting our old models into the MXNet model format if retraining is not needed. MXNet is also supported as a [backend for Keras](https://github.com/awslabs/keras-apache-mxnet), the high level neural net API, allowing us to run exactly the same Keras code developed with other frameworks with the quick MXNet backend instead.

When we have a more domain specific service in mind, where the data we are training/predicting is new but the model/task is well researched (e.g. image classification, object detection and instance segmentation), ideally we can avoid implementing a research paper from scratch and find an open source implementation to use and contribute to instead. MXNet maintains a long list of [example projects](https://github.com/apache/incubator-mxnet/tree/master/example) in its own code base for most of the popular deep learning applications/models (neural style transfer, Faster R-CNN, speech-to-text). Outside of this, MXNet has a fairly large community of open source developers that maintain MXNet versions of most of the popular and state of the art research papers in the machine learning community.

When the state of the art is just not good enough, MXNet is still an excellent choice for researching novel deep learning models. MXNet offers both symbolic (static graph) and imperative (dynamic graph) APIs allowing us to work with whichever paradigm is the most appropriate for the task. MXNet’s high-level imperative API, called [Gluon](https://mxnet.incubator.apache.org/gluon/index.html), offers a full set of plug-and-play neural network building blocks including data loaders, predefined layers and losses. This gives us the ability to save time from implementing common layers/methods for our models and spend more of our development time writing the new state of the art secret sauce in a natural Pythonic control flow. If we are working specifically with a computer vision or natural language processing task, Gluon has model toolkits that provide implementations of state-of-the-are deep learning algorithms in [GluonCV](https://gluon-cv.mxnet.io/) and [GluonNLP](http://gluon-nlp.mxnet.io/) respectively. When it comes to debugging our models during training Gluon allows us to set breakpoints to help us analyze the internals/output of our deep models, and on top of that MXNet has its own (early in development) support for writing logs out for [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) with [MXBoard](https://github.com/awslabs/mxboard).

If Python is not your data science/experimentation language of choice, MXNet also provides API’s for training in C++, Scala, Julia, Perl and R. The Scala interface also includes early support for training in a distributed [Apache Spark](https://github.com/apache/incubator-mxnet/tree/master/scala-package/spark) cluster for your big data needs.

## Inference API and Scala

After training, our models get deployed to microservices that power various Curalate applications. At Curalate, we write our microservices in Scala using the [Finatra](https://twitter.github.io/finatra/) framework. Before we switched to MXNet, we had to maintain JNA/Bridj borders between our Scala code and deep learning frameworks. This is where MXNet’s Scala API really sped up our development: any model we train in Python can be immediately loaded into our production web service framework.

Most of our deep learning microservices have a very similar data flow:

* Load an input for the deep net (in our case, typically a set of images)

* Pre-process the input data (i.e., scaling, cropping, etc)

* Perform inference on the deep net model

* Return the results to the caller

When we first started using MXNet, (version 0.11.0), the Scala API only had support for NDArrays and Modules. This was great, but we needed more functionality for higher level operations such as:

* Loading images from a network data store (in our case s3)

* Pre-processing images to match what is expected by the net (i.e., scaling, cropping, converting to raw RGB)

* Mutex locking the GPU to avoid contention

We implemented this functionality in an easy to use, high-level inference API (As of MXNet 1.2.0, the Scala API has [added support](https://mxnet.incubator.apache.org/api/scala/infer.html) for inference on images and thread management). Though our actual deployment has some Curalate-specific logic, its general design is:

<iframe src="https://medium.com/media/1fe0f4b968fd2e76585c5f54d070686f" frameborder=0></iframe>

One particularly interesting item is the mutex locking on the GPU. We achieved this by creating a singleton actor with [Akka](https://akka.io/) that acts as a gate keeper to the GPU. The actor is :

<iframe src="https://medium.com/media/e0b4bf2805967a0a6c508a364295f953" frameborder=0></iframe>

We then use Akka’s ask pattern to pass images to the InferenceActor. Not only does this let us mutex lock the GPU in a nice way, it also returns a Scala Future, which integrates well with our async Finatra patterns. The call to the actor looks like:

<iframe src="https://medium.com/media/cedcf3912a837ce7655556df276d98ba" frameborder=0></iframe>

Though we built a custom image-centric API on top of the MXNet Scala interface for our needs, MXNet treating Scala as a first class language made this extremely easy. The resulting code was minimal and elegant.

## Deployment

Continuous integration and deployment of microservices is inherently challenging, but deep learning systems add another layer of complexity due to their dependence on specialized hardware (i.e., GPUs) and native software stack (i.e., CUDA, cuDNN, and MXNet’s binary library). While our Scala-only microservices can be compiled into a war or jar and dropped into a Docker container, the story is not as simple for our deep learning services. Ideally, we’d like to deploy to these specialized environments while maintaining the agility and speed offered by continuous integration. In addition, we sometimes want to deploy to non-GPU environments (such as our development boxes, or applications where latency is OK) to save costs.

MXNet helps us solve for all of these challenges since they separate their core library from their Scala bindings API.

## MXNet Binary Packages

First, let’s look at how we can enable fast, continuous integration and deployment. MXNet’s Scala API requires two binary files be present in the environment: the core library libmxnet.so and the Java/Scala bindings libmxnet-scala.so. At Curalate, we use Jenkins to build both Docker images and AMIs for deployment in AWS. Building MXNet or its Scala bindings from source each time we need an image (for, say a new service or a different version) is time consuming and puts a serious drag on our deployment process.

To alleviate this, we build custom MXNet Debian packages that contain both the core library libmxnet.so and the Java/Scala bindings libmxnet-scala.so. Specifically, we maintain a bash script that:

* Checks out a specific version of MXNet (provided by the user)

* Compiles the MXNet base library libmxnet.so

* Compiles the MXNet Scala bindings libmxnet-scala.so

* Packages all compiled binaries into a Debian file using [checkinstall](https://help.ubuntu.com/community/CheckInstall).

* Uploads the Debian file to our s3-based repository using [deb-s3](https://github.com/krobertson/deb-s3).

We run this script inside a Docker container, so the resulting artifact is binary-compatible with the architecture of the container we compiled it in. We maintain separate repositories for different versions of Ubuntu and Debian, all of which are populated by the MXNet build script.

By compiling MXNet Debian files and maintaining our own repository, we can quickly install MXNet on any image that has access to our internal deb repository. When we set up a new service or update an existing one, our build process simply runs apt-get install libmxnet. In addition, this makes updating MXNet pretty simple: we just pass the version into our bash script and it becomes available in our repository.

## Varying Environments

Our second challenge is how to build a proper abstraction over the different hardware environments we run our deep learning services on. In production we want to run our models on GPUs for speed. This requires the host OS to have CUDA installed, and MXNet to be properly compiled against it. Often, the version of CUDA is different depending on the operating system we’re using (i.e., CUDA 8 on Ubuntu 14, CUDA 9 on Ubuntu 16). To complicate matters further, we develop on laptops without GPUs and would like to test our code in our IDEs as we build things.

To solve this, we again take advantage of MXNet’s separation of the Scala package and their core libraries. Specifically, the MXNet Scala gets packaged into mxnet-core.jar, which depends on the library libmxnet-scala.so being installed on the host operating system. So long as the MXNet version is maintained, we can package *just* the mxnet-core.jar with our services, and let the host operating system decide what stack is compiled in libmxnet-scala.so and libmxnet.so.

In other words, we again leverage our bash script from above to build multiple MXNet Debian files: one that is compiled for a GPU environment, and one for a CPU environment. We also compile a CPU environment for OSX which we install on our dev boxes via [homebrew](https://brew.sh/). The build.sbt files in our Scala projects only bring in mxnet-core, so no binary code is included in our service builds.

This flexibility allows us to create one service artifact in Scala, and run it in different environments.

## Conclusion

MXNet provides Curalate with the flexibility needed to research and build cutting edge deep learning systems extremely quickly. The Python interface has the flexibility necessary to explore research ideas while executing extremely quickly on modern hardware. For productization, Scala is treated as a first class language, providing us with an API that enables us to use MXNet in our already existing microservice architecture. Finally, the separation between the Scala API and the binary packages allow us to deploy to multiple CPU and GPU environments without recompiling or repackaging our Scala code.

*Read about more cool stuff we make at [http://engineering.curalate.com/](http://engineering.curalate.com/)*
