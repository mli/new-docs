
# Getting Started with SageMaker

Training and deploying machine learning models can sometimes be a laborious task. You need to select the model architecture, search over its hyperparameters, keep track of your experiments and make sure the results generalize once deployed. Not only are all these steps time-consuming, they also can be a serious obstacle when trying new ideas and projects.

Amazon SageMaker helps you get through all of this within a Jupyter Notebook instance — all you need is a few lines of code. And of course, an [AWS account](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html) / [Free Tier](https://aws.amazon.com/sagemaker/pricing/).

## The High Level Python Library

As first-timers to SageMaker, we would like the experience to be as smooth as possible. Fortunately, the folks at Amazon Web Services have implemented a high level Python library that abstracts most of the complexities involved in training and deploying a machine learning model, while still providing us with enough flexibility.

Our goal in this post is to train a Convolutional Neural Network on CIFAR-10 and deploy it in the cloud. To use SageMaker’s API, we need a training script. This script can be written in most of the major Deep Learning [frameworks](https://docs.aws.amazon.com/sagemaker/latest/dg/supported-versions.html): TensorFlow, PyTorch, Chainer and MXNet. We will use MXNet’s Gluon API due to its simplicity and good performance (note that it is also possible to use the Module API). The following diagram shows the three-step process we will go through: fitting the training data, deploying the trained model and predicting new inputs.

![Overview of SageMaker’s process](https://cdn-images-1.medium.com/max/2000/1*-sRbw0fWyuFu9gH3vjQc1g.png)*Overview of SageMaker’s process*

### Fitting the model

Let’s start with the basic imports needed for the project:

    import mxnet as mx
    import sagemaker
    from sagemaker.mxnet import MXNet as MXNetEstimator

The last line imports an MXNet-specific class of SageMaker’s *Estimator*. Estimators encapsulate everything you need in order to train your model. They consist of:

* methods and functions wrapping your training code,

* a [Docker](https://www.docker.com/) image used as the environment for training your model.

Using *MXNetEstimator *instead of a generic *Estimator* class means we do not need to explicitly provide a Docker image as a [default image](https://github.com/aws/sagemaker-mxnet-containers/tree/master/docker/1.1.0/final) will be used. it is also possible to set-up a [customized image](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb). In our case, we can simply use the provided one and move on to loading the data.

    mx.test_utils.get_cifar10() # Downloads Cifar-10 dataset to ./data

    sagemaker_session = sagemaker.Session()
    inputs = sagemaker_session.upload_data(path='data/cifar',
                                           key_prefix='data/cifar10')

In this code snippet, the SageMaker *Session* instance is used to upload the CIFAR-10 dataset to an S3 bucket which will be accessed by our *MXNetEstimator*.

We can now start training the model. To do so, we will instantiate the *MXNetEstimator *and use the provided *fit* method.

    estimator = MXNetEstimator(entry_point='train.py', 
                               role=sagemaker.get_execution_role(),
                               train_instance_count=1, 
                               train_instance_type='ml.p3.2xlarge',
                               hyperparameters={'batch_size': 1024, 
                                                'epochs': 30})
    estimator.fit(inputs)

The *MXNetEstimator *requires an entry point: our training script. This Python file must take a specific format, which we will describe in detail in the following section (Defining your Python script). We also need to specify our AWS role. Amongst [other things](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html), the AWS role will determine what kind of EC2 instances we can access. In this snippet, we request one *ml.p3.2xlarge* EC2 instance which contains a GPU for accelerated computing. Finally, we can specify our hyperparameters in a dictionary that will be used by the training script in the Python file.

As shown in the overview diagram, calling the *fit* method will launch the requested EC2 instance as a training environment(green rectangle), set-up the Docker image on it and start our training loop. Once this is done, we can deploy the model with the *deploy* method.

### Deploying the model

    predictor = estimator.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')

When we call the *deploy* method, SageMaker uploads the trained model to an S3 bucket and creates the prediction environment. It also returns a *Predictor*: this is the interface we will use to predict new inputs. As we see in the code snippet, it is possible to deploy our model on a different kind of instance than the one it has been trained on. This is useful as we don’t necessarily require a GPU to make predictions.

Let’s see how our predictor performs on a few pictures from CIFAR-10. We can do this with the *predict* method.

    for i, img in enumerate(image_data):
        response = predictor.predict(img)
        print('image {}: class: {}'.format(i, int(response)))

Let’s visualize these results a bit more closely:

![](https://cdn-images-1.medium.com/max/2000/1*MfEt5Yafmxj08Vvd8CwbDw.png)

The predictor seems to perform well! Once a deployed model is no longer needed, we can simply close it down:

    estimator.delete_endpoint()

In summary, we have seen how we can *fit* an MXNetEstimator, *deploy* it and then use *predict* on new inputs. All of these methods rely on our Python script— let’s have a look at it now.

## Defining your Python script

Your training loop and prediction code will be nested inside SageMaker’s library, as such, it should follow a certain format. We should define these four functions:

* *train *: a function to implement your training loop,

* *save* : a function to save your model,

* *model_fn *: a function to load your model,

* *transform_fn *: once your model is deployed, this function will be used to predict new inputs.

Only the first two functions will be used during the training phase. Let’s look at what *train* and *save* should look like (a full version of the code is available [here](https://github.com/mklissa/sagemaker_tutorial/blob/master/basics/cifar10_singlegpu.ipynb)).

<iframe src="https://medium.com/media/80c365e19895223a274f40653af0b3b9" frameborder=0></iframe>

What goes inside the *train* function is not much different than a regular training script in MXNet: you set your hyperparameters, load the data, define the model and start the training loop. To make it work, SageMaker provides you with the information you need: the location of the data (through *channel_input_dirs*) and the hyperparameters dictionary. Importantly, the *train* function should return your trained network. The *save* function is even simpler: you receive as input the trained model as well as the directory in which to save its parameters. These will later be accessed by SageMaker.

The last two functions in the Python script are used to deploy your model. Let’s see what should be contained in them:

<iframe src="https://medium.com/media/7b2c4e4773755955c5747089f220514e" frameborder=0></iframe>

The *model_fn* function will simply load the model while *transform_fn* will process the new predictions. We notice that the *inputs* variable in *transform_fn *is originally in JSON format and we need to convert it to MXNet’s NDArray format. This comes from the fact that prediction requests inside SageMaker are communicated through HTTPS. Likewise, we also need to convert our network’s outputs back to JSON so it can be used properly by the downstream code.

As we can see, the format our Python script must follow is neither too restrictive nor too complicated to implement. That being said, SageMaker can provide us with much more information if required. This can give you the opportunity to train on multiple instances or multi-GPU instances, have multiple training files and perform wide hyper-parameters searches. Interestingly enough, the process doesn’t get much more complicated than what we have seen.

While our current implementation gets to reasonable accuracy, a more complete use of SageMaker’s framework can achieve 94% accuracy on the CIFAR-10 dataset, all within 10 minutes. If this seems interesting to you, please check out our next blog.

PS: The notebook and the python scripts are available to be used as-is on this [repository](https://github.com/mklissa/sagemaker_tutorial/blob/master/basics/cifar10_singlegpu.ipynb).
