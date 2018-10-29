
# ONNX Model Zoo: Developing a face recognition application with ONNX models

Authors: Ankit Khedia, Abhinav Sharma, Hagay Lupesko

Editors: [Aaron Markham](undefined), [Thomas Delteil](undefined)

![](https://cdn-images-1.medium.com/max/2472/1*foSA83lClYcnAFOH6-onAQ.png)

Today, Amazon Web Services (AWS), Facebook and Microsoft are pleased to announce that the [Open Neural Network Exchange (ONNX) model zoo](https://github.com/onnx/models) is publicly available. [ONNX](https://onnx.ai) is an open standard format for deep learning models that enables interoperability between deep learning frameworks such as Apache MXNet, Caffe2, Microsoft Cognitive Toolkit, and PyTorch. ONNX model zoo enables developers to easily and quickly get started with deep learning using any framework supporting ONNX.

In this blogpost, we will be demonstrating how to use the [ONNX Model Zoo](https://github.com/onnx/models) to do inference with MXNet as a backend. We have chosen the model from this paper: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) by Deng et al. to showcase the usage of Model Zoo.

## ONNX Model Zoo

The ONNX community maintains the [Model Zoo](https://github.com/onnx/models), a collection of pre-trained state-of-the-art models in deep learning, available in the ONNX format. The coolest thing about the models is that they can be used with any [framework supporting ONNX](http://onnx.ai/supported-tools).

Accompanying each model are [Jupyter](http://jupyter.org/) notebooks for model training and also running inference. The notebooks are written in Python and include links to the training dataset as well as references to the original paper that describes the model architecture. The models come with a validation script to verify accuracy. The ONNX Model Zoo is unique since the contributions originate from various frameworks and hence it lives up to the vision of making interoperability easier in AI.

## Apache MXNet

[Apache MXNet](https://mxnet.apache.org) is an open-source deep learning framework used to train, and deploy deep neural networks. It is scalable, allowing for fast model training, and supports a flexible programming model and multiple languages (C++, Python, Julia, Clojure, JavaScript, R, Scala). MXNet supports both imperative and symbolic programming, which makes it easier for developers that are used to imperative programming to get started with deep learning. It also makes it easier to track, debug, save checkpoints, modify hyperparameters such as learning rate or perform early stopping. Apache MXNet supports the ONNX format for both import and export.

## ArcFace Model

The model learns distinctive features of faces and produces embeddings for input face images. For each face image, the model produces a fixed length embedding vector, like a digital fingerprint of the face. The vectors generated from different images of the same person have a higher similarity than those from different persons.

These embeddings can be used to understand the degree of similarity between two faces using metrics like cosine similarity. For the sake of demonstration, we considered a score of 0.5 or higher to be a match, i.e. the images are of the same person. The diagram below explains the workflow.

![ArcFace model workflow for measuring similarity between two faces](https://cdn-images-1.medium.com/max/2292/1*Um9zsmUt09N2jMcGEFvDhg.png)*ArcFace model workflow for measuring similarity between two faces*

## Part-1 Setting up the environment

First, we set up an environment by installing the required packages. The code was created on Ubuntu 16.04 for running inference. You can get almost all of the below by using DLAMI on EC2 instance.

    sudo apt-get install protobuf-compiler libprotoc-dev 
    pip install onnx
    pip install mxnet-mkl --pre -U 
    pip install numpy
    pip install matplotlib
    pip install opencv-python
    pip install easydict
    pip install scikit-image

Next we downloaded a few scripts, pre-trained ArcFace ONNX model and other face detection models required for preprocessing.

<iframe src="https://medium.com/media/07f5475529d000251a8fc6d489927db2" frameborder=0></iframe>

## Part-2 Loading ONNX Models into MXNet

Here, we load the ONNX model into MXNet symbols and params. It defines a model structure using the symbol object and binds parameters to the model. The models can be loaded into any other framework as long as it supports ONNX, like Caffe, CNTK and [other frameworks mentioned in official ONNX website](http://onnx.ai/supported-tools).

<iframe src="https://medium.com/media/afefbf23b29f5e4aedb9d2c9429d2664" frameborder=0></iframe>

## Part-3 Input pre-processing

Every model in the ONNX Model Zoo comes with pre-processing steps. It is an important requirement to get easily started with a given model. We used the preprocessing steps available in the provided [inference notebook](https://github.com/abhinavs95/model-zoo/blob/master/models/face_recognition/ArcFace/arcface_inference.ipynb) to preprocess the input to the models. The preprocessing involves first finding a bounding box for the face using a face detector model (MtcnnDetector model used here for detecting faces and generating bounding box coordinates) and then generating aligned face images. The whole preprocessing has been wrapped up in get_input() function in the [inference notebook](https://github.com/abhinavs95/model-zoo/blob/master/models/face_recognition/ArcFace/arcface_inference.ipynb).

Next, we perform the preprocessing steps on the two images before passing them through the model.

<iframe src="https://medium.com/media/1f56c7da78717cae9bceaae027912d4d" frameborder=0></iframe>

## Input images set 1

![Input images (set 1)](https://cdn-images-1.medium.com/max/2880/1*37J9CzsKHTkHSiqugqm9aA.png)*Input images (set 1)*

After running the preprocessing step on the two input images, we got the following output:

## Preprocessed images set 1

![Preprocessed images (set 1)](https://cdn-images-1.medium.com/max/2880/1*w6LwSqHJqYOlKrLkaVTQ7A.png)*Preprocessed images (set 1)*

## Part-4 Using pre-processed images with the model

After preprocessing, we passed the aligned faces through our model and generate embeddings for the input images.

<iframe src="https://medium.com/media/c756ff937d9c9d4f7433a7acd327ce17" frameborder=0></iframe>

The output corresponds to embeddings of the face in the input image and is a vector of size 512x1.

## Part-5 Post-processing steps

Once the embeddings are obtained, we compute their cosine similarity. Cosine similarity is a measure of similarity between two non-zero vectors that measures the cosine of the angle between them. This provides information about how close the vectors are in that space. It is equal to the dot product when the vectors are normalized.

sim = np.dot(out1, out2.T)

We obtained a similarity of 0.769120 between the two images which shows that there is high degree of match and the images are probably of the same person.

We also tried to calculate similarity scores between images of two different individuals.

## Input images set 2

![Input images (set 2)](https://cdn-images-1.medium.com/max/2880/1*fyb6PKyMKtFbJV6V5B1KdQ.png)*Input images (set 2)*

## Preprocessed images set 2

![Preprocessed images (set 2)](https://cdn-images-1.medium.com/max/2880/1*W_V5CZaesScaWsuZTn727g.png)*Preprocessed images (set 2)*

Here, we obtained a similarity score of -0.045413 between the two images (of famous personalities Diego Maradona and Edsger Djikstra) which shows that it is highly unlikely to be a match and both are images of different individuals.

We also experimented with images of Barak Obama with and without occluded face to understand how robust the network is to variations of the same face.

## Input images set 3

![Input images (set 3)](https://cdn-images-1.medium.com/max/2880/1*xP_iFFSeUtPeek812qNl1g.png)*Input images (set 3)*

## Preprocessed images set 3

![Preprocessed images (set 3)](https://cdn-images-1.medium.com/max/2880/1*bxIA2iML6oIBGBzGilBReQ.png)*Preprocessed images (set 3)*

We still managed to obtain a similarity score of 0.795367 between the two images. The model was able to identify this is the same person, despite having a slight occlusion on one of the images— showing robustness to minor changes, impressive!

Such model can be used in various use cases like facial feature based clustering, automated verification and face recognition.

## What’s next?

In this blogpost we have shown how you can use a pre-trained model from the ONNX Model Zoo. To go further, you can take advantage of the ONNX Model Zoo to do transfer learning and [fine-tuning](https://mxnet.incubator.apache.org/tutorials/onnx/fine_tuning_gluon.html) as all the layers information is available in the training scripts. We invite you to help grow the Model Zoo and contribute your own models to the ONNX community! To get started, pick any models with the [contribute](https://github.com/onnx/models/blob/master/contribute.md) link under the Description column from [ONNX Model Zoo](https://github.com/onnx/models). The links point to a page containing guidelines for adding a new model to the zoo. We look forward to your contributions!

## References

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

[InsightFace](https://github.com/deepinsight/insightface)

[ONNX Model Zoo](https://github.com/onnx/models/tree/master/models/face_recognition/ArcFace)

[Dataset](http://msceleb.org) : [MS-Celeb-1M: A Dataset and Benchmark for Large Scale Face Recognition](https://www.microsoft.com/en-us/research/publication/ms-celeb-1m-dataset-benchmark-large-scale-face-recognition-2/) (licensed for non-commercial research only)

Images used in this blog post are for demonstration purposes only, and are all licensed for non-commercial reuse.
