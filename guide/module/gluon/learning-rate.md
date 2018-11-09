# Handle learning rate


## Learning Rate Schedules

Setting the learning rate for stochastic gradient descent (SGD) is crucially important when training neural networks because it controls both the speed of convergence and the ultimate performance of the network. One of the simplest learning rate strategies is to have a fixed learning rate throughout the training process. Choosing a small learning rate allows the optimizer find good solutions, but this comes at the expense of limiting the initial speed of convergence. Changing the learning rate over time can overcome this tradeoff.

Schedules define how the learning rate changes over time and are typically specified for each epoch or iteration (i.e. batch) of training. Schedules differ from adaptive methods (such as AdaDelta and Adam) because they:

* change the global learning rate for the optimizer, rather than parameter-wise learning rates
* don't take feedback from the training process and are specified beforehand

In this tutorial, we visualize the schedules defined in `mx.lr_scheduler`, show how to implement custom schedules and see an example of using a schedule while training models. Since schedules are passed to `mx.optimizer.Optimizer` classes, these methods work with both Module and Gluon APIs.


```python
%matplotlib inline
from __future__ import print_function
import math
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import numpy as np
```

```python
def plot_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()
```

### Schedules

In this section, we take a look at the schedules in `mx.lr_scheduler`. All of these schedules define the learning rate for a given iteration, and it is expected that iterations start at 1 rather than 0. So to find the learning rate for the 100th iteration, you can call `schedule(100)`.

#### Stepwise Decay Schedule

One of the most commonly used learning rate schedules is called stepwise decay, where the learning rate is reduced by a factor at certain intervals. MXNet implements a `FactorScheduler` for equally spaced intervals, and `MultiFactorScheduler` for greater control. We start with an example of halving the learning rate every 250 iterations. More precisely, the learning rate will be multiplied by `factor` _after_ the `step` index and multiples thereafter. So in the example below the learning rate of the 250th iteration will be 1 and the 251st iteration will be 0.5.


```python
schedule = mx.lr_scheduler.FactorScheduler(step=250, factor=0.5)
schedule.base_lr = 1
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/factor.png) <!--notebook-skip-line-->


Note: the `base_lr` is used to determine the initial learning rate. It takes a default value of 0.01 since we inherit from `mx.lr_scheduler.LRScheduler`, but it can be set as a property of the schedule. We will see later in this tutorial that `base_lr` is set automatically when providing the `lr_schedule` to `Optimizer`. Also be aware that the schedules in `mx.lr_scheduler` have state (i.e. counters, etc) so calling the schedule out of order may give unexpected results.

We can define non-uniform intervals with `MultiFactorScheduler` and in the example below we halve the learning rate _after_ the 250th, 750th (i.e. a step length of 500 iterations) and 900th (a step length of 150 iterations). As before, the learning rate of the 250th iteration will be 1 and the 251th iteration will be 0.5.


```python
schedule = mx.lr_scheduler.MultiFactorScheduler(step=[250, 750, 900], factor=0.5)
schedule.base_lr = 1
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/multifactor.png) <!--notebook-skip-line-->


#### Polynomial Schedule

Stepwise schedules and the discontinuities they introduce may sometimes lead to instability in the optimization, so in some cases smoother schedules are preferred. `PolyScheduler` gives a smooth decay using a polynomial function and reaches a learning rate of 0 after `max_update` iterations. In the example below, we have a quadratic function (`pwr=2`) that falls from 0.998 at iteration 1 to 0 at iteration 1000. After this the learning rate stays at 0, so nothing will be learnt from `max_update` iterations onwards.


```python
schedule = mx.lr_scheduler.PolyScheduler(max_update=1000, base_lr=1, pwr=2)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/polynomial.png) <!--notebook-skip-line-->


Note: unlike `FactorScheduler`, the `base_lr` is set as an argument when instantiating the schedule.

And we don't evaluate at `iteration=0` (to get `base_lr`) since we are working with schedules starting at `iteration=1`.

#### Custom Schedules

You can implement your own custom schedule with a function or callable class, that takes an integer denoting the iteration index (starting at 1) and returns a float representing the learning rate to be used for that iteration. We implement the Cosine Annealing Schedule in the example below as a callable class (see `__call__` method).


```python
class CosineAnnealingSchedule():
    def __init__(self, min_lr, max_lr, cycle_length):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def __call__(self, iteration):
        if iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
            return adjusted_cycle
        else:
            return self.min_lr


schedule = CosineAnnealingSchedule(min_lr=0, max_lr=1, cycle_length=1000)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/cosine.png) <!--notebook-skip-line-->


## Using Schedules

While training a simple handwritten digit classifier on the MNIST dataset, we take a look at how to use a learning rate schedule during training. Our demonstration model is a basic convolutional neural network. We start by preparing our `DataLoader` and defining the network.

As discussed above, the schedule should return a learning rate given an (1-based) iteration index.


```python
# Use GPU if one exists, else use CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# MNIST images are 28x28. Total pixels in input layer is 28x28 = 784
num_inputs = 784
# Clasify the images into one of the 10 digits
num_outputs = 10
# 64 images in a batch
batch_size = 64

# Load the training data
train_dataset = mx.gluon.data.vision.MNIST(train=True).transform_first(transforms.ToTensor())
train_dataloader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)

# Build a simple convolutional network
def build_cnn():
    net = nn.HybridSequential()
    with net.name_scope():
        # First convolution
        net.add(nn.Conv2D(channels=10, kernel_size=5, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        # Second convolution
        net.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        # Flatten the output before the fully connected layers
        net.add(nn.Flatten())
        # First fully connected layers with 512 neurons
        net.add(nn.Dense(512, activation="relu"))
        # Second fully connected layer with as many neurons as the number of classes
        net.add(nn.Dense(num_outputs))
        return net

net = build_cnn()
```

We then initialize our network (technically deferred until we pass the first batch) and define the loss.


```python
# Initialize the parameters with Xavier initializer
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
# Use cross entropy loss
softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
```

We're now ready to create our schedule, and in this example we opt for a stepwise decay schedule using `MultiFactorScheduler`. Since we're only training a demonstration model for a limited number of epochs (10 in total) we will exaggerate the schedule and drop the learning rate by 90% after the 4th, 7th and 9th epochs. We call these steps, and the drop occurs _after_ the step index. Schedules are defined for iterations (i.e. training batches), so we must represent our steps in iterations too.


```python
steps_epochs = [4, 7, 9]
# assuming we keep partial batches, see `last_batch` parameter of DataLoader
iterations_per_epoch = math.ceil(len(train_dataset) / batch_size)
# iterations just before starts of epochs (iterations are 1-indexed)
steps_iterations = [s*iterations_per_epoch for s in steps_epochs]
print("Learning rate drops after iterations: {}".format(steps_iterations))
```



```python
schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.1)
```

**We create our `Optimizer` and pass the schedule via the `lr_scheduler` parameter.** In this example we're using Stochastic Gradient Descent.


```python
sgd_optimizer = mx.optimizer.SGD(learning_rate=0.03, lr_scheduler=schedule)
```

And we use this optimizer (with schedule) in our `Trainer` and train for 10 epochs. Alternatively, we could have set the `optimizer` to the string `sgd`, and pass a dictionary of the optimizer parameters directly to the trainer using `optimizer_params`.


```python
trainer = mx.gluon.Trainer(params=net.collect_params(), optimizer=sgd_optimizer)
```


```python
num_epochs = 10
# epoch and batch counts starting at 1
for epoch in range(1, num_epochs+1):
    # Iterate through the images and labels in the training data
    for batch_num, (data, label) in enumerate(train_dataloader, start=1):
        # get the images and labels
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # Ask autograd to record the forward pass
        with mx.autograd.record():
            # Run the forward pass
            output = net(data)
            # Compute the loss
            loss = softmax_cross_entropy(output, label)
        # Compute gradients
        loss.backward()
        # Update parameters
        trainer.step(data.shape[0])

        # Show loss and learning rate after first iteration of epoch
        if batch_num == 1:
            curr_loss = mx.nd.mean(loss).asscalar()
            curr_lr = trainer.learning_rate
            print("Epoch: %d; Batch %d; Loss %f; LR %f" % (epoch, batch_num, curr_loss, curr_lr))
```



We see that the learning rate starts at 0.03, and falls to 0.00003 by the end of training as per the schedule we defined.

### Manually setting the learning rate: Gluon API only

When using the method above you don't need to manually keep track of iteration count and set the learning rate, so this is the recommended approach for most cases. Sometimes you might want more fine-grained control over setting the learning rate though, so Gluon's `Trainer` provides the `set_learning_rate` method for this.

We replicate the example above, but now keep track of the `iteration_idx`, call the schedule and set the learning rate appropriately using `set_learning_rate`. We also use `schedule.base_lr` to set the initial learning rate for the schedule since we are calling the schedule directly and not using it as part of the `Optimizer`.


```python
net = build_cnn()
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.1)
schedule.base_lr = 0.03
sgd_optimizer = mx.optimizer.SGD()
trainer = mx.gluon.Trainer(params=net.collect_params(), optimizer=sgd_optimizer)

iteration_idx = 1
num_epochs = 10
# epoch and batch counts starting at 1
for epoch in range(1, num_epochs + 1):
    # Iterate through the images and labels in the training data
    for batch_num, (data, label) in enumerate(train_dataloader, start=1):
        # get the images and labels
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # Ask autograd to record the forward pass
        with mx.autograd.record():
            # Run the forward pass
            output = net(data)
            # Compute the loss
            loss = softmax_cross_entropy(output, label)
        # Compute gradients
        loss.backward()
        # Update the learning rate
        lr = schedule(iteration_idx)
        trainer.set_learning_rate(lr)
        # Update parameters
        trainer.step(data.shape[0])
        # Show loss and learning rate after first iteration of epoch
        if batch_num == 1:
            curr_loss = mx.nd.mean(loss).asscalar()
            curr_lr = trainer.learning_rate
            print("Epoch: %d; Batch %d; Loss %f; LR %f" % (epoch, batch_num, curr_loss, curr_lr))
        iteration_idx += 1
```



Once again, we see the learning rate start at 0.03, and fall to 0.00003 by the end of training as per the schedule we defined.

## Learning Rate Finder

Setting the learning rate for stochastic gradient descent (SGD) is crucially important when training neural network because it controls both the speed of convergence and the ultimate performance of the network. Set the learning too low and you could be twiddling your thumbs for quite some time as the parameters update very slowly. Set it too high and the updates will skip over optimal solutions, or worse the optimizer might not converge at all!

Leslie Smith from the U.S. Naval Research Laboratory presented a method for finding a good learning rate in a paper called ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186). We implement this method in MXNet (with the Gluon API) and create a 'Learning Rate Finder' which you can use while training your own networks. We take a look at the central idea of the paper, cyclical learning rate schedules, in the ['Advanced Learning Rate Schedules'](https://mxnet.incubator.apache.org/tutorials/gluon/learning_rate_schedules_advanced.html) tutorial.

### Simple Idea

Given an initialized network, a defined loss and a training dataset we take the following steps:

1. Train one batch at a time (a.k.a. an iteration)
2. Start with a very small learning rate (e.g. 0.000001) and slowly increase it every iteration
3. Record the training loss and continue until we see the training loss diverge

We then analyse the results by plotting a graph of the learning rate against the training loss as seen below (taking note of the log scales).

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_finder/finder_plot_w_annotations.png" width="500px"/>

As expected, for very small learning rates we don't see much change in the loss as the parameter updates are negligible. At a learning rate of 0.001, we start to see the loss fall. Setting the initial learning rate here is reasonable, but we still have the potential to learn faster. We observe a drop in the loss up until 0.1 where the loss appears to diverge. We want to set the initial learning rate as high as possible before the loss becomes unstable, so we choose a learning rate of 0.05.

### Epoch to Iteration

Usually, our unit of work is an epoch (a full pass through the dataset) and the learning rate would typically be held constant throughout the epoch. With the Learning Rate Finder (and cyclical learning rate schedules) we are required to vary the learning rate every iteration. As such we structure our training code so that a single iteration can be run with a given learning rate. You can implement Learner as you wish. Just initialize the network, define the loss and trainer in `__init__` and keep your training logic for a single batch in `iteration`.


```python
import mxnet as mx

# Set seed for reproducibility
mx.random.seed(42)

class Learner():
    def __init__(self, net, data_loader, ctx):
        """
        :param net: network (mx.gluon.Block)
        :param data_loader: training data loader (mx.gluon.data.DataLoader)
        :param ctx: context (mx.gpu or mx.cpu)
        """
        self.net = net
        self.data_loader = data_loader
        self.ctx = ctx
        # So we don't need to be in `for batch in data_loader` scope
        # and can call for next batch in `iteration`
        self.data_loader_iter = iter(self.data_loader)
        self.net.initialize(mx.init.Xavier(), ctx=self.ctx)
        self.loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .001})

    def iteration(self, lr=None, take_step=True):
        """
        :param lr: learning rate to use for iteration (float)
        :param take_step: take trainer step to update weights (boolean)
        :return: iteration loss (float)
        """
        # Update learning rate if different this iteration
        if lr and (lr != self.trainer.learning_rate):
            self.trainer.set_learning_rate(lr)
        # Get next batch, and move context (e.g. to GPU if set)
        data, label = next(self.data_loader_iter)
        data = data.as_in_context(self.ctx)
        label = label.as_in_context(self.ctx)
        # Standard forward and backward pass
        with mx.autograd.record():
            output = self.net(data)
            loss = self.loss_fn(output, label)
        loss.backward()
        # Update parameters
        if take_step: self.trainer.step(data.shape[0])
        # Set and return loss.
        self.iteration_loss = mx.nd.mean(loss).asscalar()
        return self.iteration_loss

    def close(self):
        # Close open iterator and associated workers
        self.data_loader_iter.shutdown()
```

We also adjust our `DataLoader` so that it continuously provides batches of data and doesn't stop after a single epoch. We can then call `iteration` as many times as required for the loss to diverge as part of the Learning Rate Finder process. We implement a custom `BatchSampler` for this, that keeps returning random indices of samples to be included in the next batch. We use the CIFAR-10 dataset for image classification to test our Learning Rate Finder.


```python
from multiprocessing import cpu_count
from mxnet.gluon.data.vision import transforms

transform = transforms.Compose([
    # Switches HWC to CHW, and converts to `float32`
    transforms.ToTensor(),
    # Channel-wise, using pre-computed means and stds
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

dataset = mx.gluon.data.vision.datasets.CIFAR10(train=True).transform_first(transform)

class ContinuousBatchSampler():
    def __init__(self, sampler, batch_size):
        self._sampler = sampler
        self._batch_size = batch_size

    def __iter__(self):
        batch = []
        while True:
            for i in self._sampler:
                batch.append(i)
                if len(batch) == self._batch_size:
                    yield batch
                    batch = []

sampler = mx.gluon.data.RandomSampler(len(dataset))
batch_sampler = ContinuousBatchSampler(sampler, batch_size=128)
data_loader = mx.gluon.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=cpu_count())
```

### Implementation

With preparation complete, we're ready to write our Learning Rate Finder that wraps the `Learner` we defined above. We implement a `find` method for the procedure, and `plot` for the visualization. Starting with a very low learning rate as defined by `lr_start` we train one iteration at a time and keep multiplying the learning rate by `lr_multiplier`. We analyse the loss and continue until it diverges according to `LRFinderStoppingCriteria` (which is defined later on). You may also notice that we save the parameters and state of the optimizer before the process and restore afterwards. This is so the Learning Rate Finder process doesn't impact the state of the model, and can be used at any point during training.


```python
from matplotlib import pyplot as plt

class LRFinder():
    def __init__(self, learner):
        """
        :param learner: able to take single iteration with given learning rate and return loss
           and save and load parameters of the network (Learner)
        """
        self.learner = learner

    def find(self, lr_start=1e-6, lr_multiplier=1.1, smoothing=0.3):
        """
        :param lr_start: learning rate to start search (float)
        :param lr_multiplier: factor the learning rate is multiplied by at each step of search (float)
        :param smoothing: amount of smoothing applied to loss for stopping criteria (float)
        :return: learning rate and loss pairs (list of (float, float) tuples)
        """
        # Used to initialize weights; pass data, but don't take step.
        # Would expect for new model with lazy weight initialization
        self.learner.iteration(take_step=False)
        # Used to initialize trainer (if no step has been taken)
        if not self.learner.trainer._kv_initialized:
            self.learner.trainer._init_kvstore()
        # Store params and optimizer state for restore after lr_finder procedure
        # Useful for applying the method partway through training, not just for initialization of lr.
        self.learner.net.save_params("lr_finder.params")
        self.learner.trainer.save_states("lr_finder.state")
        lr = lr_start
        self.results = [] # List of (lr, loss) tuples
        stopping_criteria = LRFinderStoppingCriteria(smoothing)
        while True:
            # Run iteration, and block until loss is calculated.
            loss = self.learner.iteration(lr)
            self.results.append((lr, loss))
            if stopping_criteria(loss):
                break
            lr = lr * lr_multiplier
        # Restore params (as finder changed them)
        self.learner.net.load_params("lr_finder.params", ctx=self.learner.ctx)
        self.learner.trainer.load_states("lr_finder.state")
        return self.results

    def plot(self):
        lrs = [e[0] for e in self.results]
        losses = [e[1] for e in self.results]
        plt.figure(figsize=(6,8))
        plt.scatter(lrs, losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.yscale('log')
        axes = plt.gca()
        axes.set_xlim([lrs[0], lrs[-1]])
        y_lower = min(losses) * 0.8
        y_upper = losses[0] * 4
        axes.set_ylim([y_lower, y_upper])
        plt.show()
```


You can define the `LRFinderStoppingCriteria` as you wish, but empirical testing suggests using a smoothed average gives a more consistent stopping rule (see `smoothing`). We stop when the smoothed average of the loss exceeds twice the initial loss, assuming there have been a minimum number of iterations (see `min_iter`).


```python
class LRFinderStoppingCriteria():
    def __init__(self, smoothing=0.3, min_iter=20):
        """
        :param smoothing: applied to running mean which is used for thresholding (float)
        :param min_iter: minimum number of iterations before early stopping can occur (int)
        """
        self.smoothing = smoothing
        self.min_iter = min_iter
        self.first_loss = None
        self.running_mean = None
        self.counter = 0

    def __call__(self, loss):
        """
        :param loss: from single iteration (float)
        :return: indicator to stop (boolean)
        """
        self.counter += 1
        if self.first_loss is None:
            self.first_loss = loss
        if self.running_mean is None:
            self.running_mean = loss
        else:
            self.running_mean = ((1 - self.smoothing) * loss) + (self.smoothing * self.running_mean)
        return (self.running_mean > self.first_loss * 2) and (self.counter >= self.min_iter)
```

### Usage

Using a Pre-activation ResNet-18 from the Gluon model zoo, we instantiate our Learner and fire up our Learning Rate Finder!


```python
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
net = mx.gluon.model_zoo.vision.resnet18_v2(classes=10)
learner = Learner(net=net, data_loader=data_loader, ctx=ctx)
lr_finder = LRFinder(learner)
lr_finder.find(lr_start=1e-6)
lr_finder.plot()
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_finder/finder_plot.png)


As discussed before, we should select a learning rate where the loss is falling (i.e. from 0.001 to 0.05) but before the loss starts to diverge (i.e. 0.1). We prefer higher learning rates where possible, so we select an initial learning rate of 0.05. Just as a test, we will run 500 epochs using this learning rate and evaluate the loss on the final batch. As we're working with a single batch of 128 samples, the variance of the loss estimates will be reasonably high, but it will give us a general idea. We save the initialized parameters for a later comparison with other learning rates.


```python
learner.net.save_params("net.params")
lr = 0.05

for iter_idx in range(500):
    learner.iteration(lr=lr)
    if ((iter_idx % 100) == 0):
        print("Iteration: {}, Loss: {:.5g}".format(iter_idx, learner.iteration_loss))
print("Final Loss: {:.5g}".format(learner.iteration_loss))
```


We see a sizable drop in the loss from approx. 2.7 to 1.2.

And now we have a baseline, let's see what happens when we train with a learning rate that's higher than advisable at 0.5.


```python
net = mx.gluon.model_zoo.vision.resnet18_v2(classes=10)
learner = Learner(net=net, data_loader=data_loader, ctx=ctx)
learner.net.load_params("net.params", ctx=ctx)
lr = 0.5

for iter_idx in range(500):
    learner.iteration(lr=lr)
    if ((iter_idx % 100) == 0):
        print("Iteration: {}, Loss: {:.5g}".format(iter_idx, learner.iteration_loss))
print("Final Loss: {:.5g}".format(learner.iteration_loss))
```


We still observe a fall in the loss but aren't able to reach as low as before.

And lastly, we see how the model trains with a more conservative learning rate of 0.005.


```python
net = mx.gluon.model_zoo.vision.resnet18_v2(classes=10)
learner = Learner(net=net, data_loader=data_loader, ctx=ctx)
learner.net.load_params("net.params", ctx=ctx)
lr = 0.005

for iter_idx in range(500):
    learner.iteration(lr=lr)
    if ((iter_idx % 100) == 0):
        print("Iteration: {}, Loss: {:.5g}".format(iter_idx, learner.iteration_loss))
print("Final Loss: {:.5g}".format(learner.iteration_loss))
```



Although we get quite similar results to when we set the learning rate at 0.05 (because we're still in the region of falling loss on the Learning Rate Finder plot), we can still optimize our network faster using a slightly higher rate.

### Wrap Up

Give Learning Rate Finder a try on your current projects, and experiment with the different learning rate schedules found in the tutorials [here](https://mxnet.incubator.apache.org/tutorials/gluon/learning_rate_schedules.html) and [here](https://mxnet.incubator.apache.org/tutorials/gluon/learning_rate_schedules_advanced.html).


## Advanced Learning Rate Schedules

Given the importance of learning rate and the learning rate schedule for training neural networks, there have been a number of research papers published recently on the subject. Although many practitioners are using simple learning rate schedules such as stepwise decay, research has shown that there are other strategies that work better in most situations. We implement a number of different schedule shapes in this tutorial and introduce cyclical schedules.

See the "Learning Rate Schedules" tutorial for a more basic overview of learning rates, and an example of how to use them while training your own models.


```python
%matplotlib inline
import copy
import math
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
```

```python
def plot_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()
```

### Custom Schedule Shapes

#### (Slanted) Triangular

While trying to push the boundaries of batch size for faster training, [Priya Goyal et al. (2017)](https://arxiv.org/abs/1706.02677) found that having a smooth linear warm up in the learning rate at the start of training improved the stability of the optimizer and lead to better solutions. It was found that a smooth increases gave improved performance over stepwise increases.

We look at "warm-up" in more detail later in the tutorial, but this could be viewed as a specific case of the **"triangular"** schedule that was proposed by [Leslie N. Smith (2015)](https://arxiv.org/abs/1506.01186). Quite simply, the schedule linearly increases then decreases between a lower and upper bound. Originally it was suggested this schedule be used as part of a cyclical schedule but more recently researchers have been using a single cycle.

One adjustment proposed by [Jeremy Howard, Sebastian Ruder (2018)](https://arxiv.org/abs/1801.06146) was to change the ratio between the increasing and decreasing stages, instead of the 50:50 split. Changing the increasing fraction (`inc_fraction!=0.5`) leads to a **"slanted triangular"** schedule. Using `inc_fraction<0.5` tends to give better results.


```python
class TriangularSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        inc_fraction: fraction of iterations spent in increasing stage (float)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length*self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle
```

We look an example of a slanted triangular schedule that increases from a learning rate of 1 to 2, and back to 1 over 1000 iterations. Since we set `inc_fraction=0.2`, 200 iterations are used for the increasing stage, and 800 for the decreasing stage. After this, the schedule stays at the lower bound indefinitely.


```python
schedule = TriangularSchedule(min_lr=1, max_lr=2, cycle_length=1000, inc_fraction=0.2)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/adv_triangular.png) <!--notebook-skip-line-->


#### Cosine

Continuing with the idea that smooth decay profiles give improved performance over stepwise decay, [Ilya Loshchilov, Frank Hutter (2016)](https://arxiv.org/abs/1608.03983) used **"cosine annealing"** schedules to good effect. As with triangular schedules, the original idea was that this should be used as part of a cyclical schedule, but we begin by implementing the cosine annealing component before the full Stochastic Gradient Descent with Warm Restarts (SGDR) method later in the tutorial.


```python
class CosineAnnealingSchedule():
    def __init__(self, min_lr, max_lr, cycle_length):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def __call__(self, iteration):
        if iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
            return adjusted_cycle
        else:
            return self.min_lr
```

We look at an example of a cosine annealing schedule that smoothing decreases from a learning rate of 2 to 1 across 1000 iterations. After this, the schedule stays at the lower bound indefinietly.


```python
schedule = CosineAnnealingSchedule(min_lr=1, max_lr=2, cycle_length=1000)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/adv_cosine.png) <!--notebook-skip-line-->


### Custom Schedule Modifiers

We now take a look some adjustments that can be made to existing schedules. We see how to add linear warm-up and its compliment linear cool-down, before using this to implement the "1-Cycle" schedule used by [Leslie N. Smith, Nicholay Topin (2017)](https://arxiv.org/abs/1708.07120) for "super-convergence". We then look at cyclical schedules and implement the original cyclical schedule from [Leslie N. Smith (2015)](https://arxiv.org/abs/1506.01186) before finishing with a look at ["SGDR: Stochastic Gradient Descent with Warm Restarts" by Ilya Loshchilov, Frank Hutter (2016)](https://arxiv.org/abs/1608.03983).

Unlike the schedules above and those implemented in `mx.lr_scheduler`, these classes are designed to modify existing schedules so they take the argument `schedule` (for initialized schedules) or `schedule_class` when being initialized.

#### Warm-Up

Using the idea of linear warm-up of the learning rate proposed in ["Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" by Priya Goyal et al. (2017)](https://arxiv.org/abs/1706.02677), we implement a wrapper class that adds warm-up to an existing schedule. Going from `start_lr` to the initial learning rate of the `schedule` over `length` iterations, this adjustment is useful when training with large batch sizes.


```python
class LinearWarmUp():
    def __init__(self, schedule, start_lr, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        start_lr: learning rate used at start of the warm-up (float)
        length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.length:
            return iteration * (self.finish_lr - self.start_lr)/(self.length) + self.start_lr
        else:
            return self.schedule(iteration - self.length)
```

As an example, we add a linear warm-up of the learning rate (from 0 to 1 over 250 iterations) to a stepwise decay schedule. We first create the `MultiFactorScheduler` (and set the `base_lr`) and then pass it to `LinearWarmUp` to add the warm-up at the start. We can use `LinearWarmUp` with any other schedule including `CosineAnnealingSchedule`.


```python
schedule = mx.lr_scheduler.MultiFactorScheduler(step=[250, 750, 900], factor=0.5)
schedule.base_lr = 1
schedule = LinearWarmUp(schedule, start_lr=0, length=250)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/adv_warmup.png) <!--notebook-skip-line-->


#### Cool-Down

Similarly, we could add a linear cool-down period to our schedule and this is used in the "1-Cycle" schedule proposed by [Leslie N. Smith, Nicholay Topin (2017)](https://arxiv.org/abs/1708.07120) to train neural networks very quickly in certain circumstances (coined "super-convergence"). We reduce the learning rate from its value at `start_idx` of `schedule` to `finish_lr` over a period of `length`, and then maintain `finish_lr` thereafter.


```python
class LinearCoolDown():
    def __init__(self, schedule, finish_lr, start_idx, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        finish_lr: learning rate used at end of the cool-down (float)
        start_idx: iteration to start the cool-down (int)
        length: number of iterations used for the cool-down (int)
        """
        self.schedule = schedule
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.start_lr = copy.copy(self.schedule)(start_idx)
        self.finish_lr = finish_lr
        self.start_idx = start_idx
        self.finish_idx = start_idx + length
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.start_idx:
            return self.schedule(iteration)
        elif iteration <= self.finish_idx:
            return (iteration - self.start_idx) * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        else:
            return self.finish_lr
```

As an example, we apply learning rate cool-down to a `MultiFactorScheduler`. Starting the cool-down at iteration 1000, we reduce the learning rate linearly from 0.125 to 0.001 over 500 iterations, and hold the learning rate at 0.001 after this.


```python
schedule = mx.lr_scheduler.MultiFactorScheduler(step=[250, 750, 900], factor=0.5)
schedule.base_lr = 1
schedule = LinearCoolDown(schedule, finish_lr=0.001, start_idx=1000, length=500)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/adv_cooldown.png) <!--notebook-skip-line-->


##### 1-Cycle: for "Super-Convergence"

So we can implement the "1-Cycle" schedule proposed by [Leslie N. Smith, Nicholay Topin (2017)](https://arxiv.org/abs/1708.07120) we use a single and symmetric cycle of the triangular schedule above (i.e. `inc_fraction=0.5`), followed by a cool-down period of `cooldown_length` iterations.


```python
class OneCycleSchedule():
    def __init__(self, start_lr, max_lr, cycle_length, cooldown_length=0, finish_lr=None):
        """
        start_lr: lower bound for learning rate in triangular cycle (float)
        max_lr: upper bound for learning rate in triangular cycle (float)
        cycle_length: iterations between start and finish of triangular cycle: 2x 'stepsize' (int)
        cooldown_length: number of iterations used for the cool-down (int)
        finish_lr: learning rate used at end of the cool-down (float)
        """
        if (cooldown_length > 0) and (finish_lr is None):
            raise ValueError("Must specify finish_lr when using cooldown_length > 0.")
        if (cooldown_length == 0) and (finish_lr is not None):
            raise ValueError("Must specify cooldown_length > 0 when using finish_lr.")

        finish_lr = finish_lr if (cooldown_length > 0) else start_lr
        schedule = TriangularSchedule(min_lr=start_lr, max_lr=max_lr, cycle_length=cycle_length)
        self.schedule = LinearCoolDown(schedule, finish_lr=finish_lr, start_idx=cycle_length, length=cooldown_length)

    def __call__(self, iteration):
        return self.schedule(iteration)
```

As an example, we linearly increase and then decrease the learning rate from 0.1 to 0.5 and back over 500 iterations (i.e. single triangular cycle), before reducing the learning rate further to 0.001 over the next 750 iterations (i.e. cool-down).


```python
schedule = OneCycleSchedule(start_lr=0.1, max_lr=0.5, cycle_length=500, cooldown_length=750, finish_lr=0.001)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/adv_onecycle.png) <!--notebook-skip-line-->


#### Cyclical

Originally proposed by [Leslie N. Smith (2015)](https://arxiv.org/abs/1506.01186), the idea of cyclically increasing and decreasing the learning rate has been shown to give faster convergence and more optimal solutions. We implement a wrapper class that loops existing cycle-based schedules such as `TriangularSchedule` and `CosineAnnealingSchedule` to provide infinitely repeating schedules. We pass the schedule class (rather than an instance) because one feature of the `CyclicalSchedule` is to vary the `cycle_length` over time as seen in [Ilya Loshchilov, Frank Hutter (2016)](https://arxiv.org/abs/1608.03983) using `cycle_length_decay`. Another feature is the ability to decay the cycle magnitude over time with `cycle_magnitude_decay`.


```python
class CyclicalSchedule():
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1, cycle_magnitude_decay=1, **kwargs):
        """
        schedule_class: class of schedule, expected to take `cycle_length` argument
        cycle_length: iterations used for initial cycle (int)
        cycle_length_decay: factor multiplied to cycle_length each cycle (float)
        cycle_magnitude_decay: factor multiplied learning rate magnitudes each cycle (float)
        kwargs: passed to the schedule_class
        """
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwargs = kwargs

    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length
        cycle_offset = iteration - idx + cycle_length

        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
        return schedule(cycle_offset) * self.magnitude_decay**cycle_idx
```

As an example, we implement the triangular cyclical schedule presented in ["Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith (2015)](https://arxiv.org/abs/1506.01186). We use slightly different terminology to the paper here because we use `cycle_length` that is twice the 'stepsize' used in the paper. We repeat cycles, each with a length of 500 iterations and lower and upper learning rate bounds of 0.5 and 2 respectively.


```python
schedule = CyclicalSchedule(TriangularSchedule, min_lr=0.5, max_lr=2, cycle_length=500)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/adv_cyclical.png) <!--notebook-skip-line-->


And lastly, we implement the scheduled used in ["SGDR: Stochastic Gradient Descent with Warm Restarts" by Ilya Loshchilov, Frank Hutter (2016)](https://arxiv.org/abs/1608.03983). We repeat cosine annealing schedules, but each time we halve the magnitude and double the cycle length.


```python
schedule = CyclicalSchedule(CosineAnnealingSchedule, min_lr=0.01, max_lr=2,
                            cycle_length=250, cycle_length_decay=2, cycle_magnitude_decay=0.5)
plot_schedule(schedule)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/lr_schedules/adv_sgdr.png) <!--notebook-skip-line-->


**_Want to learn more?_** Checkout the "Learning Rate Schedules" tutorial for a more basic overview of learning rates found in `mx.lr_scheduler`, and an example of how to use them while training your own models.
