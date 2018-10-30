
# 94% accuracy on CIFAR-10 in 10 minutes with Amazon SageMaker

A model’s test accuracy provides crucial information: it is an estimate of how well the model would perform when released in the real world on unseen data. However, it is not the whole picture. In a recent blogpost, OpenAI showed that the amount of compute required to train start-of-the-art models has progressed in an exponential way in the last few years. This has many consequences, one of them being a potential slow down of the research process — that is, unless you can afford it.

To shift the focus from final accuracy to other metrics of a model, Stanford recently held an end-to-end deep learning benchmark called [DAWNBench](https://dawn.cs.stanford.edu/benchmark/index.html). In one of the categories, contestants were asked to come up with models that would compete on shortest training time while specifying a minimum threshold for accuracy.

The top entry for training time on CIFAR-10 used distributed training on multi-GPU to achieve 94% in a little less than 3 minutes! This was achieved by [fast.ai](http://www.fast.ai/). However, the [code](https://github.com/fastai/imagenet-fast) provided takes a bit more time to go through.

In this blogpost, we want to get close to the best results on CIFAR-10, while at the same time reducing the complexity of the code. We will do so through Amazon SageMaker’s efficient API.

## Distributed training on SageMaker

In our previous [blogpost](https://medium.com/apache-mxnet/getting-started-with-sagemaker-ebe1277484c9), we have seen the basics of how to use SageMaker. If you are new to this, it is a recommended read! Once done, implementing distributed training with SageMaker will be as simple as rewriting a few lines of our single-GPU code. Once again, we will be using MXNet as our deep learning framework of choice as it scales well with distributed training.

### SageMaker’s Notebook instance

The first lines of code in the Notebook follow the same logic as for single-GPU:

    import mxnet as mx
    import sagemaker
    from sagemaker.mxnet import MXNet as MXNetEstimator

    mx.test_utils.get_cifar10() # Downloads Cifar-10 dataset to ./data

    sagemaker_session = sagemaker.Session()
    inputs = sagemaker_session.upload_data(path='./data',
                                           key_prefix='data/cifar10')

We simply import the necessary modules, create a SageMaker session and upload the dataset to an S3 bucket. The lines we need to modify will be the ones defining our MXNetEstimator, i.e. the class used to launch the distributed training job. Let’s see how this should look like:

    type_instance = 'ml.p3.8xlarge'
    num_instance = 1
    source_dir = 'training_code'

    estimator = MXNetEstimator(entry_point='multitrain.py',
                               source_dir=source_dir
                               role=sagemaker.get_execution_role(),
                               train_instance_count=num_instance, 
                               train_instance_type=type_instance,
                               hyperparameters={'batch_size': 128, 
                                                'epochs': 10})
    estimator.fit(inputs)

When doing distributed training, it is possible to choose between multi-GPU or multi-instance (or even both). The *type_instance* variable will specify what type of [instance](https://aws.amazon.com/sagemaker/pricing/instance-types/) we will need. In our case, *ml.p3.8xlarge *indicates an EC2 instance with four Tesla V100 GPUs, therefore choosing multi-GPU training. It could also be possible to change the *num_instance *variable to a higher integer and have multi-instance distributed training instead. Due to the more complex communication pipeline for multi-instance, multi-GPU is significantly faster. In our experience, using multi-GPU scales almost-linearly for 2 GPUs, however, going past that number provides very modest speed-ups.

That is all we have to do in order to launch distributed training jobs! SageMaker will take care of the rest by creating the requested instances and setting up optimal communication between them. As usual, we need to provide a valid Python script that contains our training loop — let’s check it out.

## Adjusting the entry point script

It is not enough to request a multi-GPU instance, we still need to modify our training algorithm accordingly to make a good use of the hardware. As such, we need to modify the *train* function contained in the entry point script. An important variable we will define is the [key-value store](https://mxnet.incubator.apache.org/api/python/kvstore/kvstore.html#mxnet.kvstore.create) (KVStore), as it is used for data synchronization over multiple devices and instances, mostly to optimize the model’s parameters updates.

![A Server contains a copy of the model’s weights. It passes those weights to Workers who then perform gradient updates on GPU.](https://cdn-images-1.medium.com/max/2512/1*sZOT4kiXAID4Sazol8w6dg.png)*A Server contains a copy of the model’s weights. It passes those weights to Workers who then perform gradient updates on GPU.*

We will encounter three different values that we can assign to kvstore:

* *local*: CPU memory will be used to update the weights’ gradients,

* *device*: GPU memory will be used to update the weights’ gradients, and it will use GPU peer-to-peer communication in multi-GPU settings,

* *dist_device_sync*:* *same configuration as *device,* except it is used for multi-instance GPU training (corresponding to the previous figure).

Let’s have a look at how we implement this in our training code (a full version of the code is available [here](https://github.com/mklissa/sagemaker_tutorial/blob/master/multigpu/cifar10_multigpu.ipynb)):

<iframe src="https://medium.com/media/45f48b35402af646d508325e54df2d7c" frameborder=0></iframe>

Compared to our previous single-GPU implementation, the *train* function receives a few more parameters. This is convenient as we can choose the parameters that are important to our implementation and exclude the rest.

Since we do distributed training, we will need to know the *hosts, current_host*, *num_gpus* and *num_cpus*. These parameters let us define the *kvstore *variable, which will be passed to the SGD optimizer (line 40) for an efficient distribution of the update operations.

The extra parameters also let us define *part_index*, which is passed to the data loading function *get_train_data *(line 32). As we can see in the next figure, in a case of multi-instance training, the dataset will be divided by the number of instances and *part_index *will give each instance the correct part it should pick. Lastly, we need to specify the correct context (*ctx*) that will be used by the training loop’s *split_and_load* function (line 44): this will divide the batch between the available GPUs.

![Distributed training on multi-instances and multi-GPU](https://cdn-images-1.medium.com/max/2000/1*VMnx69Cy-_E81gIk2wPGBQ.png)*Distributed training on multi-instances and multi-GPU*

Those are the only necessary modifications we need to address for distributed training. These modifications will insure we use all GPUs/instances, however it does not guarantee that we will max-out on GPU utilization. To do so, we need to linearly scale the batch size with the number of GPUs and the number of instances. As this seems to be an easy fix, it actually leads to more trouble.

Indeed, it is an increasingly known fact by deep learning researchers that larger batch sizes can lead to degradation of the accuracy performance. To understand why, we need to look at the scale of the random fluctuations introduced by stochastic gradient descent (as opposed to gradient descent).

![Formula for random noise “g” induced by SGD](https://cdn-images-1.medium.com/max/2896/1*0emzNlFRmb9808CoZEdHpg.png)*Formula for random noise “g” induced by SGD*

In this formula, ϵ is the learning rate, N is the training set size and B is the batch size. As [Smith et al.](https://arxiv.org/abs/1711.00489) suggest, there is an optimal batch size that creates just enough noise to escape sharp minima, which generalize poorly. Therefore, to compensate for the increase in batch size it is possible to increase the learning rate by the same factor. For example, an original learning rate of 0.1 would become 0.2 for distributed learning on 2 GPUs. However, this technique has its limits: augmenting the learning beyond a certain point might lead to divergence. To learn more about how to set your learning rate, please read Thom Lane’s [blogpost](https://medium.com/@thom.e.lane/24c2ac7d4fe4) on the Learning Rate Finder.

Our goal was to understand how to use SageMaker for distributed training. We have seen that the changes we have to apply to SageMaker’s API are minimal: we either need to specify an instance that contains multiple GPUs or choose to launch multiple instances. Our Python training script is a bit more elaborate, as we have to set up the right values for *kv_store*, *part_index*, *ctx *and adjust the *batch_size*. However, the few lines of code that define those variables can be used as-is for all of your distributed training.

We also had another goal: we wanted to get close to some of the best results on the DAWNBench competition. Now that we are all set, let’s see how fast we can get to 94% accuracy on CIFAR-10.

![Results on CIFAR-10](https://cdn-images-1.medium.com/max/2552/1*biNw4xYxlW0ZGAPM44iIjA.png)*Results on CIFAR-10*

We achieved 94% in under 10 minutes! That is a good start. However, to improve our results we need more specialized techniques, such as better hyperparameter search. To see how we can get there, please read our next blogpost.
