
# Clojure Package for MXNet

One of the strengths of MXNet is its multi-language support. With the shared backend written in C, you can train, use, and scale your deep learning models with the language you are comfortable in. As of the v1.3 release, the family of language bindings for MXNet has grown from Python, Scala, C++, Julia, Perl, and R to include Clojure.

The Clojure MXNet package opens up modern deep learning, and flexible and efficient GPU computing to Clojure users everywhere. To give you a taste of what is available, we’ll take a tour to highlight a few things you can do.

## Image Recognition

You can load state-of-art [pre-trained models](https://mxnet.incubator.apache.org/model_zoo/) in MXNet and quickly do predictions on images.

The function below gets the [RESNET-152](https://arxiv.org/pdf/1512.03385v1.pdf) pre-trained model and loads it into a module. It also loads the [*synset.txt](http://data.mxnet.io/models/imagenet-11k/synset.txt)* labels for the classification categories. The function accepts an image url parameter and runs the image through the model for classification returning the top 5 probabilities with their labels.

<iframe src="https://medium.com/media/063ff205c3bf5347455f7e7192ceddaf" frameborder=0></iframe>

Let’s see what happens when we give it the image of this tabby cat.

![[Dustin Warrington](https://www.flickr.com/people/firewall/) — [Flickr](https://www.flickr.com/photos/firewall/91092531/)](https://cdn-images-1.medium.com/max/2000/1*TbhIG3rD_sN6HY2Es1d1fA.jpeg)*[Dustin Warrington](https://www.flickr.com/people/firewall/) — [Flickr](https://www.flickr.com/photos/firewall/91092531/)*

<iframe src="https://medium.com/media/e16c6baf3178841e26ee0a828a67c9a2" frameborder=0></iframe>

We can see that the top predication accurately chooses the tabby. For the full code, you can check out the [pre-trained models example](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/pre-trained-models) in the github repo.

You can also create your own models and train them with Module API. The Clojure package provides a clean way to compose your own layers. The following code constructs a network of fully connected and activation layers that can be used for training on MNIST handwritten digits.

<iframe src="https://medium.com/media/b572ddb393f780a94185c5ba2e978406" frameborder=0></iframe>

You can explore the[ Clojure Module documentation](https://mxnet.incubator.apache.org/api/clojure/module.html) to learn more about training and predicting.

## Generative Models

Generative models are available for Clojure. There is full support for GANs *(Generative Adversarial Networks)*. A great example of this is using the [MNIST handwritten digits example](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/gan). From a training set of digits, you can see the program gradually start to generate more and more realistic images as the training progresses.

![](https://cdn-images-1.medium.com/max/2000/1*y9xvF-lcCf1L9Dws45cfNQ.gif)

## Natural Language Processing

With the Clojure package, you can also do deep learning with text. There is an example to get you started using [convolutional neural networks for sentence classification using movie reviews](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/cnn-text-classification).

You can also explore using RNNs (Recurrent Neural Networks) to generate text from a corpus with this [example](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/rnn). The program starts with no knowledge of language or language rules. It just trains on the text of a corpus of Obama’s speeches by taking one character and trying to predict what character comes next. Gradually, over the course of many epochs, it learns how to generate sentences. When given some starter text of *“The joke” *it produces something that is surprisingly good.
> The joke before them prepared for five years ago, we only hear a chance to lose our efforts and they made striggling procedural deficit at the city between a politics in the efforts on the Edmund Pett

## Wrap Up

The Clojure API for MXNet opens up exciting opportunities for the Clojure community to get involved with deep learning in the language you love. Dive in and get started today with the [online ](http://mxnet.incubator.apache.org/api/clojure/index.html)and the [project documentation](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package).
