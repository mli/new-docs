
# Leveling up on SageMaker

In our previous blogpost, we have gone through the basics of SageMaker. Most of the complexities were hidden from us due to the efficiently implemented API, which lets us train and deploy a model in just a few lines. However, it is possible to go beyond the basic configurations to gain better control and insight on our models. To do so, we will cover the following key components:

* Accessible models and endpoints

* Logs and metrics

* Training in local mode

The good news is that we will continue using SageMaker’s API throughout this tutorial — keeping things simple, while giving us greater flexibility.

## Accessible models and endpoints

SageMaker’s API works directly from a Jupyter Notebook where you can launch a training job and deploy your model. However, the whole process is not directly tied to the notebook instance itself. This is good news if you need to train a model for days or if your interactive session with the notebook is lost, but you then need to know how to fetch an existing training job and how to connect to an existing deployed model (i.e. an endpoint).

### Fetching a training job

In order to retrieve an existing training job, we will need to know its name. When launching a training job, SageMaker will by default assign it a name along the line of “sagemaker-mxnet-2018–07–09–22–59–29–027”. It is also possible to specify a name through the *base_job_name *parameter given to MXNetEstimator. In our case, we haven’t specified anything. A possible way to retrieve a job’s name is going through SageMaker’s website.

![Accessing SageMaker’s training jobs through the web interface.](https://cdn-images-1.medium.com/max/2648/1*Wy1WgnmpBvwuAqJy9usfsA.png)*Accessing SageMaker’s training jobs through the web interface.*

We can now proceed to fetch the training process inside a notebook:

    estimator = MXNetEstimator(entry_point='train.py', 
                               role=sagemaker.get_execution_role(),
                               train_instance_count=1, 
                               train_instance_type='ml.p3.2xlarge')
    estimator.attach('sagemaker-mxnet-2018-07-10-18-29-40-566')

The instantiation parameters passed to MXNetEstimator are not important in this case as the job has already been launched. Once *attach *is called, the complete output of your algorithm, from the start to the most recent line, will be printed.

The previous situation assumes we have a training job running, or that we have kept a record of a completed job. However, we may only have a local copy of the final saved model (i.e. a model.tar.gz file). Nevertheless, it is still possible to load and deploy it in the following way:

    from sagemaker.mxnet.model import MXNetModel

    sagemaker_model = MXNetModel(model_data='path/to/model.tar.gz',
                                 role=sagemaker.get_execution_role(),
                                 entry_point='train.py')

    predictor = sagemaker_model.deploy(initial_instance_count=1,
                                       instance_type='ml.m4.xlarge')

The model’s parameters can either be located on the local disk or on an S3 bucket.

### Connecting to an existing endpoint

Given that we have deployed a model in the past, we might be interested in retrieving its endpoint (possibly to another interface than the SageMaker notebook, i.e. a web interface). Once again, all we need is the name. When calling the Estimator’s *fit* method, the model will be saved and be made accessible on SageMaker’s website:

![Accessing SageMaker’s models through the web interface.](https://cdn-images-1.medium.com/max/2648/1*puFuN9RlIyjvj0jMjEQtiw.png)*Accessing SageMaker’s models through the web interface.*

The saved model will take the same name as the training job’s name. We now only need two lines to connect to the existing endpoint and proceed to predictions.

    from sagemaker.mxnet import MXNetPredictor

    predictor = MXNetPredictor(‘sagemaker-mxnet-2018–06–29–2254–45–509’)

## Logs and metrics

When training a job, everything that is being logged onto the notebook instance is also being logged onto [CloudWatch](https://aws.amazon.com/cloudwatch/). This means that we can easily recover and read through the logs of past jobs. To do this, we need to go through CloudWatch’s web interface.

![Accessing your training logs through CloudWatch](https://cdn-images-1.medium.com/max/2954/1*IO3WCzvYxBjxHjDx4mvesQ.png)*Accessing your training logs through CloudWatch*

Logs provide us with important information, but we could get an even more precise picture through the collected [metrics](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/sagemaker-metricscollected.html).

![Accessing metrics through CloudWatch](https://cdn-images-1.medium.com/max/3150/1*MMHqyCL3Dk6rFhagH1z3vw.png)*Accessing metrics through CloudWatch*

In the previous figure we have searched for a training job’s metrics, however it also is possible to get information about our endpoints. As an example we might be interested in knowing how many requests are being sent to a particular endpoint.

![Training job’s metric graph from CloudWatch](https://cdn-images-1.medium.com/max/4148/1*mN5rM7EZ2GfMPnBMxF1Jtw.png)*Training job’s metric graph from CloudWatch*

CloudWatch will therefore provide us with crucial information, especially if we want to make sure our endpoint has enough resources for the amount of requests it receives. Fortunately enough, there is a clever way to deal with fluctuations in demand, that is through [auto-scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html). To turn on this feature, we need to proceed in the following way:

![Selecting auto-scaling for better management of our endpoints](https://cdn-images-1.medium.com/max/2962/1*0sbbgyNzkbOV5sty9nC27Q.png)*Selecting auto-scaling for better management of our endpoints*

## Training in local mode

The delay taken for starting a job is not negligible, notably when one wants to make sure the code still runs smoothly after a few minor modifications. The people at SageMaker have made things easier for us through the *local mode*, which reduces the waiting time to zero. As the name suggests, the training code will be launched on the same instance as the one SageMaker’s notebook is running on. We have a choice between *local_cpu* and *local_gpu* (for cases when our notebook instance has an integrated GPU). We will need to upload the following [script](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_cifar10/setup.sh) (which installs the local dependencies) to our notebook instance and run it through the notebook.

    !/bin/bash setup.sh # Run the script in the notebook

    instance_type='local_gpu' # Choose local training (CPU or GPU)

    m = MXNet('train.py', 
              role=role, 
              train_instance_count=1, 
              train_instance_type=instance_type, 
              hyperparameters={'batch_size': 1024, 
                               'epochs': 50})

This feature has helped us numerous times in the past, hopefully this will help you too. However, it must be noted that multi-instance is not currently supported while training in local mode.

## Conclusion

We have reviewed three different ways in which we can expand our control over SageMaker. We have seen that SageMaker’s API lets us retrieve training jobs, saved models and endpoints in just a few lines while CloudWatch provides us with important information regarding each of these instances. We have also seen that training in local mode lets us prototype and debug in a much more efficient way. Some of these advantages come at the cost of an increased complexity, which we believe should be manageable — if you experience any difficulties while trying them out, please let us know.
