
# Relation Networks for Visual Question Answering using MXNet Gluon

Overview

Visual Question Answering (VQA) is a multi-modal task relating text and images through captions or a questionnaire. For example, with a picture of a busy highway, there could be a question: “How many red cars are there?” or “Are there more motorbikes than cars?”. It is a very challenging task since it requires high-level understanding of both the text and the image and the relationships between them. Today we will go over the *Relational Network* architecture proposed by Deep Mind in 2016, which beat all previous methods by a significant margin.

![An example of VQA task](https://cdn-images-1.medium.com/max/2000/1*SR_TCaUo-G8JJ2S-VMiGrg.png)*An example of VQA task*

The main idea of the relational network is not only conceptually easy to understand but also is widely applicable to other domains.

## Data description

In this article, we will use a simplified version of CLEVR: sort-of-CLEVR³, which contains 10,000 images with 20 questions. An image contains randomly generated objects from two shape candidates(square or circle) and 6 color candidates(red, green, blue, orange, gray and yellow). For a given image, 10 of those questions are relational questions and another 10s are non-relational ones.

![Source: [Relational Networks](https://github.com/kimhc6028/relational-networks)](https://cdn-images-1.medium.com/max/2000/1*iQWh2LqjQkGLj-Xf5ox0ZA.png)*Source: [Relational Networks](https://github.com/kimhc6028/relational-networks)*

I.e., with the sample image shown, we can generate non-relational questions like:

1. Is the green object placed on the left side of the image? => yes

1. Is the orange object placed on top of the image? => no

And relational questions:

1. What is the shape of the object closest to the red object? => square

1. What is the shape of the object farthest from the orange object? => circle

For a detailed description of the data, visit this [github repo](https://github.com/kimhc6028/relational-networks).

## Network Architecture

The general formula for the relational network can be written as follows:

![](https://cdn-images-1.medium.com/max/2000/1*G6xCPYaMetFkTCAcEwz-hw.png)

First, we convert the data to **an object(**O*i* and O*j*). Then, after applying a function *g* to every combination of object pairs, we add them up before passing them through another function *f*.

The most important thing in a relational network is to consider an object regardless of the domain. Therefore, we can perform learning within only one network for different domains. Therefore,

The Process of relational network in VQA problem is as follow:

1. Generate feature maps

* Question : Embedding using LSTM

* Image : CNN (4 convolutional layers)

2. Apply RN(Relational Network) steps

* Generate object pair with question : We make all possible combinations for objects extracted from question and image.

* Add coordination position information for particle image feature.

* Apply the first 4 layers MLP(function* g)*

* Apply element-wise sum for the output of function *g*

* Apply the next two layers MLP(function *f*)

![An end-to-end relational networks¹](https://cdn-images-1.medium.com/max/2420/1*RlhLUYEUHRFPtt4TMamYFg.png)*An end-to-end relational networks¹*

For better understanding, let’s examine a simple example. Suppose that there are three 3x3 CNN feature maps and one 1x4 query feature vector. Then, we end up with 9*9= 81 combinations of feature vectors. Please note that the first dimension will be inflated 81 times more than the original batch size, since we created 81 new data pairs. After applying a function *g*, those inflated batches will be reduced to the original batch size by element-wise summation before being passed to a function* f*. Assuming that each individual image object has a length of 3 and the question object has length of 4 , we can depict the architecture as below:

![An example of data structure for RN](https://cdn-images-1.medium.com/max/2000/1*RQ0a2Lbx64NI9SVH1oFuwA.png)*An example of data structure for RN*

The coordinate position vector is a kind of vector to provide pixel position information. The structure of coordinate position is as follow:

    **def** cvt_coord(i):
        **return** [(i/5-2)/2., (i%5-2)/2.]

![An example of coordinate position](https://cdn-images-1.medium.com/max/2000/1*wfBW7oOcP96DaI5PxtOdog.png)*An example of coordinate position*

## Code

Let‘s’ implement a relational network model with Gluon. For data generation, we used [this code](https://github.com/kimhc6028/relational-networks/blob/master/sort_of_clevr_generator.py)² to generating input data. First, we make input data using sort_of_clevr_generator.py. Before executing the generation code, we need to install the necessary packages.

    pip install opencv-python
    python sort_of_clevr_generator.py

Once you run the code you will see the following message. And then you are good to go. One of generated images is displayed in the following figure. The question is provided as a one-hot encoded feature vector. So we do not consider the LSTM-encoding part in the example code.

    building test datasets...
    building train datasets...
    saving datasets...
    datasets saved at ./data/sort-of-clevr.pickle

![The structure of Input Data](https://cdn-images-1.medium.com/max/2000/1*86KyCz689szXKUMsgLYkig.png)*The structure of Input Data*

To speed up training, we will use multiple gpus. For that you can just set the context parameter as shown below:

    GPU_COUNT = 2

    def setting_ctx(GPU_COUNT):
        if GPU_COUNT > 0 :
            ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
        else :
            ctx = [mx.cpu()]
        return ctx
            
    ctx = setting_ctx(GPU_COUNT)

Then we need to define a class for relational network. It consists of three parts: Convolution, DNN (*g*), and classifier (*f*). The convolution part, which has 4 convolutional layers, can be defined as follows:

<iframe src="https://medium.com/media/671e0e434c13bf183db932489345d25e" frameborder=0></iframe>

The classifier (function *f*), is very simple and has only 2 fully connected layers as depicted below:

<iframe src="https://medium.com/media/d8fee3d6962a902e06337015bb9d14de" frameborder=0></iframe>

The core of the architecture of the relation network can be broken down into the following steps:

1. Generate feature maps from the Convolution encoder.

1. Add coordinate information to consider pixel locations.

1. Concatenate all possible pairs of the image feature maps, and concatenate the question features. Then, apply FCN(function *g*).

1. Compute the element-wise sum to scale back to the original batch size and use Classifier(function *f*).

<iframe src="https://medium.com/media/6e2a7141b5acdf5f9e4d08025162b86e" frameborder=0></iframe>

Use adam optimizer and softmax cross entropy loss.

    #set optimizer
    trainer = gluon.Trainer(model.collect_params(),optimizer='adam')

    #define loss function
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

Now, we load the data using the **split_and_load()** function to easily use multiple GPU resource. It automatically divides the input data and allocate each subset to a different GPU.

<iframe src="https://medium.com/media/293346963648c4f818a7a3c185b58155" frameborder=0></iframe>

Implementation details can be found [here](https://github.com/seujung/relational-network-gluon/blob/master/relation_reasoning_code_use_multi_gpu.ipynb).

## Result

We achieved 99% accuracy in the no-relational problem and 91% accuracy (for test data) in the relational problem.

![The perfornamce of relational network for test data (Accuracy)](https://cdn-images-1.medium.com/max/3732/1*bFwAcnepkIkv6FquKGiRQw.png)*The perfornamce of relational network for test data (Accuracy)*

![The performance of relational network for test data (Loss)](https://cdn-images-1.medium.com/max/3752/1*0wGRW1VrqgiePLQizopFeQ.png)*The performance of relational network for test data (Loss)*

## Conclusion

* The relational network has shown great performance in the VQA domain using a simple algorithm structure.

* Gluon provides an easy and straightforward API to implement multi-gpu usage for model training.

## Reference

[1] Paper : [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)

[2] [https://deepmind.com/blog/neural-approach-relational-reasoning/](https://deepmind.com/blog/neural-approach-relational-reasoning/)

[3] [https://github.com/kimhc6028/relational-networks](https://github.com/kimhc6028/relational-networks)
