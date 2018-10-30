
# Train using Keras-MXNet and inference using MXNet Scala API



## Overview

Apache MXNet(incubating) recently announced [support for Keras 2](https://aws.amazon.com/blogs/machine-learning/apache-mxnet-incubating-adds-support-for-keras-2/), we have already talked about some of the great features in this [blog post](https://medium.com/mxnet/keras-gets-a-speedy-new-backend-with-keras-mxnet-3a853efc1d75). In this tutorial, we will walk you through one of the use cases where you can design your model, train using Keras-MXNet, and run inference in production, for example on [Spark](https://github.com/apache/incubator-mxnet/tree/master/scala-package/spark). We will use the latest [Scala Inference API](https://medium.com/apache-mxnet/scala-api-for-deep-learning-inference-now-available-with-mxnet-v1-2-bcb13235db95) released with MXNet 1.2.0.

## Train a ResNet-50 Model on CIFAR10 Dataset
> *Note: This functionality only supports models trained in **channels first** data format, which has better performance according to the [keras-mxnet performance guide](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/performance_guide.md). Change "image_data_format"to "channels_first" and set "backend" to "mxnet" in '~/.keras/keras.json'*

We will extend the work from our [release blog post](https://aws.amazon.com/blogs/machine-learning/apache-mxnet-incubating-adds-support-for-keras-2/), please follow the instruction there for [installation](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/installation.md) and [training ](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/multi_gpu_training.md)using multi GPUs. We used Keras[ example script](https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/cifar10_resnet_multi_gpu.py) to train a ResNet-50 Model on CIFAR10 Dataset .
 For installation, if you are using GPU and have cuda 9.0 or above installed, just use the following command:

    pip install mxnet-cu90
    pip install keras-mxnet

If you have not cloned our [repo](https://github.com/awslabs/keras-apache-mxnet/tree/master), you can get the training script by using:

    wget [https://raw.githubusercontent.com/awslabs/keras-apache-mxnet/master/examples/cifar10_resnet_multi_gpu.p](https://raw.githubusercontent.com/awslabs/keras-apache-mxnet/master/examples/cifar10_resnet_multi_gpu.p)y

In order to save native MXNet models, add the following line at the end of your script. What happens under the hood is we are using native MXNet Module in Keras backend for better multi-GPU support, an extra perk doing that is you can save symbols and trained params in native MXNet format. For more details on how it works, please refer to [Save MXNet model from Keras-MXNet](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/save_mxnet_model.md).

    data_names, data_shapes = save_mxnet_model(model=model, prefix='cifar10_resnet50', epoch=0)

Next, start the training script on N GPUs. (4 in our case, modify num_gpus variable accordingly based on the GPUs you have)

    python cifar10_resnet_multi_gpu.py

After training finished at 100 epochs, you will see the following output:

    Test loss: 1.13192281456
    Test accuracy: 0.8308
    MXNet Backend: Successfully exported the model as MXNet model!
    MXNet symbol file -  cifar10-resnet50-symbol.json
    MXNet params file -  cifar10-resnet50-0000.params
    

    Model input data_names and data_shapes are: 
    data_names :  ['/input_11']
    data_shapes :  [DataDesc[/input_11,(128L, 3L, 32L, 32L),float32,NCHW]]

At 100 epochs, we reached 83.08% test accuracy. Record the data_names and data_shapes values as we will use them in the next part. Note that native MXNet model consists of two files, the cifar10-resnet50-symbol.json file contains the symbolic graph and the cifar10-resnet50-000.params records the parameters of the model.

## Inference using MXNet Scala Inference API

Now it’s time to load the saved model and run inference. We will use the[ Scala Inference API](https://medium.com/apache-mxnet/scala-api-for-deep-learning-inference-now-available-with-mxnet-v1-2-bcb13235db95) released with MXNet 1.2.0. We will use the[ image classification](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier) example, and run our pre-trained model by just changing model names and input data shapes.

## Environment Setup

You should already have MXNet installed for Keras-MXNet to work, follow this [tutorial](https://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html) to setup all other prerequisites for MXNet-Scala development:

1. [Install MXNet Scala Package](https://mxnet.incubator.apache.org/install/ubuntu_setup.html#install-the-mxnet-package-for-scala)

1. [IntelliJ IDE (or alternative IDE) project setup](http://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html) with the MXNet Scala Package

## Prepare Model and Data

After setting up IntelliJ, we need to create two folders for model and inference data. For the model folder, place the two pre-trained model files there. In addition, we need to prepare a synset.txt file that contains the label ID to label name mapping. For CIFAR10 dataset, it looks like the following:

    0 airplane
    1 automobile
    2 bird
    3 cat
    4 deer
    5 dog
    6 frog
    7 horse
    8 ship
    9 truck

In side the model folder resnet-50, you will have:

    .
    └── resnet-50
        ├── cifar10-resnet50-0000.params
        ├── cifar10-resnet50-symbol.json
        └── synset.txt

For inference, we randomly saved 10 images from the CIFAR10 test dataset using a simple script in python and Keras. The output file name is in the fomat: 'x_test_index_label'. Run the following python script, and you will get the testing images.

    import keras
    import random
    import scipy.misc
    from keras.datasets import cifar10

    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    num_test = len(x_test)
    num_images = 10
    for idx in random.sample(range(0, num_test), num_images):
        label = y_test[idx][0]
        output = 'x_test_' + str(idx) + '_' + str(label) +'.jpg'
        scipy.misc.imsave(output, x_test[idx])

![Inference Test Images](https://cdn-images-1.medium.com/max/2300/1*GGXsR4sK93q9cUz58ePu7A.png)*Inference Test Images*

## Load Pre-trained MXNet Model using Scala API

Now, we only need to load the [image classifier example](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/ImageClassifierExample.scala) in IntelliJ and change a few lines (See complete modified code [here](https://gist.github.com/roywei/9e71272fcf2ccb64a81f072999ec47f1)). Remember the data_names and data_shapes from the previous section? We need to pass them for MXNet to load the model, change line [47](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/ImageClassifierExample.scala#L47), [49](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/ImageClassifierExample.scala#L49), [68](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/ImageClassifierExample.scala#L68), [70](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/ImageClassifierExample.scala#L70) accordingly in the runInferenceOnSingleImage and runInferenceOnBatchOfImage methods. Since we are training and inference on the CIFAR10 dataset (32x32 size) in channel first format, change the inputShape to (1, 3, 32, 32), and use the name '/input_11' that we got from the data_names variable in previous section.

    val inputShape = Shape(1, 3, 32, 32)
    val inputDescriptor = IndexedSeq(DataDesc("/input_11", inputShape, dType, "NCHW"))

The complete runInferenceOnSingleImage method will look like the following, same rules applies to runInferenceOnBatchOfImage.

<iframe src="https://medium.com/media/8d675977414187d74aa82d9eca8427bf" frameborder=0></iframe>

## Run Inference

To run the inference, we use the following 3 arguments:

    1. The model file prefix we saved in training section
    --model-path-prefix "absolute/path/to/your/model"
    e.g. "/Users/lawei/scalaMXNet/models/resnet-50/cifar10-resnet50"

    2. Image to be predicted
    --input-image "absolute/path/to/your/image"
    e.g. "/Users/lawei/scalaMXNet/images/x_test_202_8.jpg"

    3. Directory of images to be predicted
    --input-dir "absolute/path/to/image/directory"
    e.g. "/Users/lawei/scalaMXNet/images"

Pass these arguments in IntelliJ, and then run the inference

![Pass the arguments in IntelliJ](https://cdn-images-1.medium.com/max/2144/1*an-WK2twtgfV_xQzRGBVqg.png)*Pass the arguments in IntelliJ*

You will get the following result, most of the testing images are predicted correctly. We are showing one example been predicted as horse (squint your eyes really hard, it is really a horse!):

![Input image for inference](https://cdn-images-1.medium.com/max/2000/1*ZAuSAQ8N_Pv-NMb4U6zOqQ.png)*Input image for inference*

    Input image x_test_1584_7.jpg 
    Class with probability =Vector((7 horse,0.7975957), (3 cat,0.09379086), (4 deer,0.07512564), (0 airplane,0.032357983), (8 ship,4.789475E-4)) 

## What’s Next

With the Scala API, you can run large scale inference on Spark, check out our tutorial and blog post for more details:

1. [MXNet on Spark](https://github.com/apache/incubator-mxnet/tree/master/scala-package/spark)

1. [Blog post on MXNet and Spark on AWS EMR](https://aws.amazon.com/blogs/machine-learning/distributed-inference-using-apache-mxnet-and-apache-spark-on-amazon-emr/)

To learn more about the Scala API, read our medium blog posts series:

1. [Scala Inference API](https://medium.com/apache-mxnet/scala-api-for-deep-learning-inference-now-available-with-mxnet-v1-2-bcb13235db95)

1. [Image Classification with Scala API](https://medium.com/apache-mxnet/image-classification-with-mxnet-scala-inference-api-8ab6ce1bbccf)

1. [Object Detection with Scala API](https://medium.com/apache-mxnet/object-detection-with-mxnet-scala-inference-api-9049230c77fd)

## Citations

1. CIFAR: “Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.”

1. ResNet: “Identity Mappings in Deep Residual Networks,Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.”

1. ResNet: “Deep Residual Learning for Image Recognition, Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun, 2015”
