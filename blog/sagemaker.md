
# SageMaker Automatic Model Tuner: CIFAR10 in 3m47s on single GPU

In our previous blogpost, we trained a model to reach 94% accuracy on CIFAR10 in about 10 minutes. This was done by training the model on a multi-GPU instance with the help of SageMaker’s API, which made the whole process easy and straightforward.

Recently, there has been an increase in interest for faster convergence time on CIFAR10 and ImageNet. Just a few weeks ago, the [Tencent AI](https://arxiv.org/abs/1807.11205) lab was able to train a ResNet50 on ImageNet up to an accuracy of 75.8% in a little over 4 minutes! Here’s the catch though: they used **2048 GPUs** to get the result. While the techniques they developed to coordinate such an amount of GPUs are interesting and innovative, we can wonder how this will affect researchers that do not have access to this kind of hardware.

Therefore, in this blogpost, we propose to focus on** single GPU **training in order to converge on CIFAR10. We were able to get there in only 3min47 seconds, which, to our knowledge, is a new record for single-GPU training. We used various optimization techniques, which we will explain in this blogpost, but throughout the process we had one special tool: an automatic model tuner.

## Hyperparameter tuning

When training deep learning models, we are faced with the difficulty of choosing values for a great amount of hyperparameters: learning rate, momentum, weight decay, model architecture and more. All the while, we are using precious GPU resources to figure it out. There is a smarter way though. Given our ResNet model is reasonably slow to train (i.e. >5 minutes) and evaluate for different hyperparameter combinations, we can instead learn a secondary model (a.k.a surrogate model or meta-model) to predict the performance of the ResNet (in this case accuracy). Using this secondary model we can intelligently select our next candidate hyperparameters to evaluate; balancing both exploration and exploitation.

That is exactly what SageMaker’s Automatic Model Tuner is doing. It uses a Gaussian Process as the secondary model, and adjusts the standard sequential approach by parallelizing candidate selection. The good thing is that you don’t need to know a single thing about Gaussian Processes, as all this is done automatically in the background, but if you’re interested check out [this video](https://www.youtube.com/watch?v=vz3D36VXefI). Moreover, the API is straightforward to use once again. Let’s have a look at it.

### Defining the Search Space

We have to import the usual modules in order to use SageMaker (defined in the previous blog). To use SageMaker’s Automatic Model Tuner, we need to import the HyperparameterTuner* *object, and each of the data types we’ll want to optimize:

    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

During the tuning job, we don’t need to optimize all the hyperparameters (for example we might have prior knowledge about how to set some of them). Therefore, some hyperparameters will be varied and some will remain fixed. To do this, we define the fixed hyperparameters through the usual [MXNet estimator](https://medium.com/apache-mxnet/getting-started-with-sagemaker-ebe1277484c9).

    hyperparameters = {'batch_size': 512,
                       'epochs': 32 }

    estimator = MXNet(entry_point='run.py', 
                      source_dir=source_dir,
                      role=role, 
                      train_instance_count=1, 
                      train_instance_type='ml.p3.2xlarge',
                      framework_version='1.1.0',
                      hyperparameters=hyperparameters)

To define the variable hyperparameters, we use a separate dictionary, in which we indicate the ranges and the data types of the variables we want to search through. Given that we are using a smarter optimization technique compared to a simple grid search, you can choose a range that is wider than usual in order to explore more possibilities.

    hyperparameter_ranges = {
                            'lr': ContinuousParameter(0.2, 0.65),
                            'mom': ContinuousParameter(0.87, 0.92),    
                            'epochs':IntegerParameter(28,35),
                            'opt':CategoricalParameter(['sgd', 'nag']),
                            }

In this examples we have defined three different kind of [hyperparameters](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/):

* Continuous: e.g. the learning rate, which will vary continuously from 0.1 to 0.65

* Integer: e.g. the number of training epochs, which will vary discretely from 28 to 35,

* Categorical: e.g. the choice of optimizer, in this case ‘Stochastic Gradient Descent’ or ‘Nesterov’s Momentum’.

### Choosing what to optimize

These three categories constitute all the possible kinds of hyperparameters we can optimize. We now need to define what metric will be used to guide the model tuning process. In our case, this is the validation accuracy. We could also modify this metric to include a bonus in case of faster convergence, since this is what are aiming for in the end. However, the validation accuracy proxy seemed to work well enough to avoid adding heuristics. To let the tuning job know what metric we are interested in, we need to explicitly output its value through logs. For example, if we output the line “Epoch 15, Validation accuracy=0.78”, the HyperparameterTuner will pick up on that. Of course, we first need to define that through a [regex](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285):

    metric_definitions = [{'Name': 'Validation-accuracy',
                           'Regex': 'Validation accuracy=([0-9\\.]+)'}]

In this case, we told the tuner that we will be outputting ‘Validation accuracy=0.00’ (be careful about typos, it needs to be exactly the same). We are now ready to define our HyperparameterTuner and launch the whole process:

    tuner = HyperparameterTuner(estimator=estimator,
                         objective_metric_name='Validation-accuracy',         
                         hyperparameter_ranges=hyperparameter_ranges,
                         metric_definitions=metric_definitions,
                         max_jobs=20,
                         max_parallel_jobs=2,
                         base_tuning_job_name='my-tuning-job')
    tuner.fit(inputs)

It is interesting to notice that through the parameters max_jobs and max_parallel_jobs we are faced with a tradeoff between the expected performance and the rapidity of the tuning job. Indeed, if we choose to run many jobs in parallel, we’ll expect a speedup, but there will be fewer iterations of the tuning process, and we start to lose the benefits of the secondary model. As a rule of thumb, set the max_parallel_jobs to around 10% of the max_jobs, to allow at least 10 steps of the secondary model.

Of course, while our tuning job is running we would like to know how well it is performing. We could do so by going through the AWS web console:

![Using AWS web console to access your tuning job.](https://cdn-images-1.medium.com/max/2980/1*--B2E4bh-qGIXEod9vewHA.png)*Using AWS web console to access your tuning job.*

As we see in the figure, in a few clicks we have access to interesting information: the best training job so far, the list of training jobs, the hyperparameter configurations and more. By going through the training jobs, we can access each of the log streams and metrics (as we have seen in the previous blog).

Perhaps more interestingly, it is possible to visualize our tuning job with a few lines of code. We include all these details in the [repository](https://github.com/mklissa/sagemaker_tutorial/blob/master/hyperparameter_tuning/HyperParameterTuning.ipynb) associated with this blogpost. We can plot the timeline of the tuning job. It is interesting to notice that the distribution of datapoints starts off randomly and explores the hyperparameter space. After a few hours it seems to consistently converge to 94%, therefore exploiting what it has found about our model. This would not be the case for a grid search.

![Plot of the tuning job’s timeline. The X axis is the time of the day, each new tick representing 2 hours.](https://cdn-images-1.medium.com/max/3632/1*CS5NmB2u9xyBvneRCo09Qg.png)*Plot of the tuning job’s timeline. The X axis is the time of the day, each new tick representing 2 hours.*

As we see, this is an interactive plot which lets you get the basic information about a certain training job (i.e. the hyperparameters for that particular run). In a glance we can then pick up possible trends the tuning jobs has found for our model. We can also plot each hyperparameter with respect to the objective:

![Plot of the validation accuracy vs hyperparameter. The relationships between hyperparametesr and performance are usually complex.](https://cdn-images-1.medium.com/max/2856/1*tLR-0t4J3FONrJPqZfv0iQ.png)*Plot of the validation accuracy vs hyperparameter. The relationships between hyperparametesr and performance are usually complex.*

In some cases, there is a definite region of the hyperparameter space that leads to better results, as is the case for the learning rate and the weight decay. In other cases there is no clear preference, as we can see for the momentum. Don’t forget that we’re just looking at univariate plots here, and in reality there could be multivariable relationships too.

### Beyond hyperparameter tuning: superconvergence

While the ability to tune jobs was essential to set our new record, we also used superconvergence techniques. We will only enumerate some of them, for a full picture, the code is available on the same repository.

Perhaps the most important element was the learning rate schedule. As per [Smith](https://arxiv.org/abs/1506.01186), we will be using a cyclical learning rate, with a single cycle (a.k.a 1-Cycle policy). Our learning rate schedule will have the following shape:

![Cyclical learning rate, using a single cycle.](https://cdn-images-1.medium.com/max/2580/1*jXL23rahUo1lIIYH2nGdTw.png)*Cyclical learning rate, using a single cycle.*

Secondly, we use large batch sizes in order to speed up training. In our case, we use a batch size of 512 which is four times larger than the often seen 128. Following the [current wisdom](https://arxiv.org/abs/1711.00489) in deep learning, we should multiply the original learning rate by the same factor, i.e. 4, in order to keep the same amount of noise in the gradient updates (this noise leads to better results on the validation set). However, through extensive hyperparameters search we figured that the optimal learning rate would be around 0.2 rather than 0.4 (given that the original value was 0.1), as we can see in this figure. When you see evaluations clustering on one side of your search space range (as in the diagram below), you should extend the search range.

![Validation Accuracy vs Learning rate value: in subsequent tuning jobs, we performed a wider search, but still found that training jobs converged better around the value of 0.2 for the learning rate.](https://cdn-images-1.medium.com/max/3224/1*Yiw_Mwzbx7mM1eirHTmfHA.png)*Validation Accuracy vs Learning rate value: in subsequent tuning jobs, we performed a wider search, but still found that training jobs converged better around the value of 0.2 for the learning rate.*

More investigation would be needed, but at least in our case the findings of the [state-of-the-art](https://arxiv.org/pdf/1711.00489.pdf) do not apply. This wouldn’t be first time that the inner workings of neural networks are being questioned, as we have [recently](https://www.reddit.com/r/MachineLearning/comments/8n4eot/r_how_does_batch_normalization_help_optimization/) seen that batch normalization doesn’t work for the reasons we thought it did. This is a good reminder that there still a great discrepancy between theory and practice when it comes to deep learning: making efficient hyper-parameter search all the more important.

We have used a few more important techniques, but we leave the details to the [implementation](https://github.com/mklissa/sagemaker_tutorial/blob/master/hyperparameter_tuning/HyperParameterTuning.ipynb), which should hopefully be straightforward to understand. As a side note, the fastest result comes when the number of epochs is set to 30, however this means the number of times the jobs converge to 94% is not high. In order to get to that number on almost each run, the number of epochs should be around 33 to 35. At this number, the training time takes a little more than four minutes, which is still currently the fastest convergence time on a single GPU.

## Conclusion

Through this series of blogposts, we have seen how to use SageMaker for simple jobs, distributed training and tuning jobs. We have also seen some more advanced functionalities which give us better control for our experiments. Importantly, we have shown that SageMaker is quite a well-rounded and straightforward tool to use. Hopefully your research will be improved by it!
