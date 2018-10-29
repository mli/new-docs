
# Deploy a Smile Detector with Keras- MXNet and MXNet Model Server



## Overview

AWS recently announced the [release of the Apache MXNet backend for Keras 2](https://medium.com/apache-mxnet/keras-gets-a-speedy-new-backend-with-keras-mxnet-3a853efc1d75), which can give you up to 3x speed improvements compared to the default backend with multi-GPU. It exhibits improved performance for both training and inference. With this [release](https://github.com/awslabs/keras-apache-mxnet/releases/tag/v2.1.6), you are now able to able to export the trained [Keras (with MXNet backend)](https://github.com/awslabs/keras-apache-mxnet) model as a native MXNet model, with no dependency on a specific language binding. This new export model capability allows you to use various tools that are part of the MXNet eco-system.

In this blog post, we will be demonstrating the usage of the Keras-MXNet models with [MXNet Model Server](https://github.com/awslabs/mxnet-model-server/). We will train a model to detect smiles in images, and then host it for online inference over a web API using MXNet Model Server. MXNet Model Server is a tool for serving deep learning models, that supports [MXNet](https://github.com/apache/incubator-mxnet) and [Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx) models, and handles various aspects of model serving in production including HTTP endpoints, scalability, real-time metrics and more.

## Part 1 — Training the model with Keras-MXNet

As mentioned above, we will train a model to detect smiles. We will follow the steps mentioned in this [SmileCNN open source repository](https://github.com/kylemcdonald/SmileCNN), written by [Kyle McDonald](https://github.com/kylemcdonald) to train our model.

First, we setup the environment by installing the necessary packages. If you are using CPU based machine run:

    pip install mxnet
    pip install keras-mxnet

For GPU based machines with cuda 9.0 installed, run “pip install mxnet-cu90”
> Note: Detailed installation instructions on Keras-MXNet can be accessed [here](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/installation.md). Machines with GPU (such as AWS P or G instance types) give better training performance. This can be achieved by launching [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/)

Next, we created a fork of the SmileCNN repository, modified the notebooks and converted them into python files to fit our existing use-case for inferring using MXNet Model Server. The modified files can be accessed by cloning the following:

    git clone [https://github.com/kalyc/SmileCNN](https://github.com/kalyc/SmileCNN)
    cd SmileCNN

This repository contains three python files — dataset preparation, training and evaluation.

First, we install the needed dependencies for this package using:

    sudo python setup.py install

Then, we prepare the training data by running [dataset preparation](https://github.com/kalyc/SmileCNN/blob/master/datasetprep.py)* *file.

    python datasetprep.py

Running this file downloads the training dataset. The dataset is nearly 40MB examples of non-smiling and smiling faces. The dataset has a folder for each of positive and negative training images. This script takes these images and resizes them from *64 x 64* to *32 x 32* pixels. It then converts them into [numpy arrays](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.array.html) and updates their format for usage with Keras-MXNet.

Once the data is ready for use, we train the model using the [training](https://github.com/kalyc/SmileCNN/blob/master/train.py) file. Saving model in Keras-MXNet is currently only supported for *channels_first* data format, which is known to have a better performance according to the [Keras-MXNet performance guide](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/performance_guide.md). The Keras configuration needs to be updated to use the *channels_first *image* *data format:
> The Keras configuration file can be accessed at $HOME/.keras/keras.json

    {
     “backend”: “mxnet”,
     “image_data_format”: “channels_first”
    }

Next, we use the *to_channels_first()* function to convert the input data in *channels_first* format in the training script on [line 24](https://github.com/kalyc/SmileCNN/blob/master/train.py#L24).

    import keras as k
    X = k.utils.to_channels_first(X)

We modify the original training script to save the model and add [line 64](https://github.com/kalyc/SmileCNN/blob/master/train.py#L64):

    # Save the trained Keras Model as MXNet Model
    keras.models.save_mxnet_model(model=model, prefix='smileCNN_model')

After updating the configuration, we run the training script.

    python train.py

The training network is built based on the [mnist_cnn example](https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/mnist_cnn.py). Training this model can take a varying length of time depending on your hardware configuration. In this case, we trained the model for 100 epochs on *p2.16xlarge *on Base Deep Learning AMI. It took about 3 minutes. The P instance comes with GPU, thereby allowing a faster training time. On CPU based hardware the training can take much longer.

    Test accuracy: 0.9638663504942575

Once the training is complete, the *save_mxnet_model()* function returns the following output:

    MXNet Backend: Successfully exported the model as MXNet model!
    MXNet symbol file -  smileCNN_model-symbol.json 
    MXNet params file -  smileCNN_model-0000.params

    Model input data_names and data_shapes are:

    data_names :  ['/conv2d_1_input1']
    data_shapes :  [DataDesc[/conv2d_1_input1,(4L, 1L, 32L, 32L),float32,NCHW]]
> Note: In the above data_shapes, the first dimension represents the batch_size used for model training. You can change the batch_size for binding the module based on your inference batch_size.

The *save_mxnet_model()* function creates the *smileCNN_model-symbol* and *smileCNN_model-0000.params* files upon successfully saving the model. These files define the structure of the network and the associated weights. They essentially define the trained MXNet model. The input symbol is */conv2d_1_input1,* of shape (4L, 1, 32, 32).

The *smileCNN_model-symbol.json* and *smileCNN_model-0000.params* files are generated at the root of the directory.

We can test the accuracy of the model using the evaluation script.

    python evaluation.py

If everything is setup correctly, the model should be able to take a numpy array and predict the result as ‘smiling’.

![Evaluation Image](https://cdn-images-1.medium.com/max/2000/1*c9ACANMJw6l05ga2U_sUNw.png)*Evaluation Image*

         ('non-smiling', '------------------------###-', 'smiling')

The results above clearly show that the image is classified correctly. The evaluation script uses the model saved with Keras-MXNet and loads it for use in prediction.

## Part 2 — Inference with MXNet Model Server

Next, let’s see how we can use [MXNet Model Server (MMS)](https://github.com/awslabs/mxnet-model-server) for serving this model.

Following the [quick Start guide with MMS](https://github.com/awslabs/mxnet-model-server#quick-start), we setup MXNet Model Server on our machines.

On ubuntu run:

    sudo apt-get install protobuf-compiler libprotoc-dev
    pip install mxnet-model-server

On mac run:

    conda install -c conda-forge protobuf
    pip install mxnet-model-server

We have created a model archive directory with the name of [*keras-mms](https://github.com/kalyc/SmileCNN/tree/master/keras-mms) *in the SmileCNN repository*. *We move the saved trained model’s symbol and params files into the keras-mms directory, which would be used for hosting model inferencing on MXNet Model Server.

    cp smileCNN_model-* ./keras-mms/
    cd keras-mms/
> Note: The section below walks through the process of creating the files already present exist in the [keras-mms directory](https://github.com/kalyc/SmileCNN/tree/master/keras-mms).

The model archive directory (*keras-mms*) contains these files:

    - signature.json
    - synset.txt
    - smileCNN_model-symbol.json
    - smileCNN_model-0000.params
    - custom_service.py

In order to let MMS know which input symbol and what shape to use for inference, we use the output from *save_mxnet_model()* function and setup the *signature.json *file in the same directory as follows:

<iframe src="https://medium.com/media/945ee3267490805db639232161f23ffa" frameborder=0></iframe>

The signature specifies that the input expected for inferencing is a JPEG image. The output type is JSON. Output data shape varies between 0 and 1 as the model predicts only 2 classes as *smiling* and *non-smiling*.

We add the necessary* synset.txt* file to list labels — with one label per line as mentioned in the [MXNet-Model Server export README](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export.md). This file varies per dataset and contains the list of classes a model can predict.

For the smile detection example, the class labels are:

    non-smiling
    smiling

Then, we write a *custom_service.py* file that [defines custom preprocessing and post-processing inference](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export.md).

<iframe src="https://medium.com/media/e47f84c4cb4da42b5995d0b57bc9f3b6" frameborder=0></iframe>

Then, we generate the MMS *.model* file using the files in the *keras-mms *directory with the following command:

    mxnet-model-export --model-name smileCNN --model-path . --service-file-path custom_service.py

This generates *smileCNN.model *file. Simply put, this file is the artifact that contains all the dependencies necessary to run our model as a webservice with MMS.

Finally, we run this command to start the server:

    mxnet-model-server --models smileCNN=smileCNN.model

The model server is up and ready for use!

It is time to test the model!
> Note: All images below have been scaled up for visibility

![Test Images](https://cdn-images-1.medium.com/max/2000/1*chG6OZvrPiQtoIaX2lfXDg.png)*Test Images*

![test-1.jpg](https://cdn-images-1.medium.com/max/2000/1*6N_6spi4NhCdIBwdDtfYlg.jpeg)*test-1.jpg*

    curl -X POST [http://127.0.0.1:8080/smileCNN/predict](http://127.0.0.1:8080/cifar10_resnet/predict) -F "/conv2d_1_input1=@test-1.jpg"

Resulting prediction:

    {
        "prediction": [
            [
                {
                    "class": "smiling",
                    "probability": 1.0
                },
                {
                    "class": "non-smiling",
                    "probability": 0.0
                }
            ]
        ]
    }

And voilà! This is how we train a model using Keras with MXNet backend and run inference through MXNet Model Server.
> The inferred probabilities of the model may appear skewed as MXNet Model Server rounds those up.

To further evaluate the model, we test it with a different example.

![test-2.jpg](https://cdn-images-1.medium.com/max/2000/1*7Ij0vbORwSQrT-niXGw0BQ.png)*test-2.jpg*

    curl -X POST [http://127.0.0.1:8080/smileCNN/predict](http://127.0.0.1:8080/cifar10_resnet/predict) -F "/conv2d_1_input1=@test-2.jpg"

Resulting prediction

    {
        "prediction": [
            [
                {
                    "class": "non-smiling",
                    "probability": 1.0
                },
                {
                    "class": "smiling",
                    "probability": 0.0
                }
            ]
        ]
    }

We can see that the predictions made by this simple CNN model are already pretty good!

This is a toy model but it could have real-world applications. For example, to make sure your colleagues are always smiling when coming in the office, you can add a webcam above the office front-door and require a smile to get access to the building! :)

**Learn More **
The latest release of Keras-MXNet allows users to train large-scale models at a higher speed and export the trained model in the MXNet native format to allow inference on several platform, including MXNet-Model-Server. To learn more about Keras-MXNet — be sure to follow the [Keras-MXNet code repository](https://github.com/awslabs/keras-apache-mxnet). Further details about MXNet Model Server can be found on the [MXNet Model Server code repository](https://github.com/awslabs/mxnet-model-server). Contributions and feedback from the community are welcome!

If you liked what you read — follow the [Apache MXNet channel on Medium](https://medium.com/apache-mxnet) to be updated on the latest developments of the MXNet ecosystem. Want to do inference on Spark using Scala instead? Read our blog post on [training with Keras and using the MXNet Scala API for inference](https://medium.com/apache-mxnet/train-using-keras-mxnet-and-inference-using-mxnet-scala-api-49476a16a46a)! 
 
**Citations**

1. SmileCNN: “Kyle McDonald, 2016.”

1. SMILEsmileD: “Horni, October 2010”

Editors: [Wei Lai](undefined), [Sandeep Krishnamurthy](undefined), [Aaron Markham](undefined)
