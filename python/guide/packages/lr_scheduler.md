# Schedule Learning Rate

Setting the learning rate for SGD is crucially
important when training neural networks because it controls both the speed of
convergence and the ultimate performance of the network. The
`mxnet.lr_scheduler` module provides multiple commonly used scheduling methods
for setting the learning rate.

```{.python .input}
from IPython import display
from matplotlib import pyplot as plt
import math

from mxnet import lr_scheduler, optimizer
```

## Basic Usages

A scheduler returns a learning rate for a given iteration
count, which starts at 1. In the following example, we create a scheduler that
returns the initial learning rate 1 for the first 250 iterations, then then
halve its value for every 250 iterations.

```{.python .input}
scheduler = lr_scheduler.FactorScheduler(base_lr=1, step=250, factor=0.5)
```

Let's verify it on a few iterations.

```{.python .input}
scheduler(1), scheduler(250), scheduler(251), scheduler(501)
```

A scheduler is often passed as a argument when creating an optimizer, such as

```{.python .input}
optim = optimizer.SGD(learning_rate=0.1, lr_scheduler=scheduler)
```

Note that, when specifying the initial learning rate through the `learning_rate`
argument, it will overwrite the `base_lr` for the scheduler.

```{.python .input}
optim.lr_scheduler.base_lr
```

## Commonly Used Scheduler

Next, we will visualize several commonly used
schedulers. We first define a function to plot the learning rate for the first
1000 iterations.

```{.python .input}
def plot(scheduler, num_iterations=1000):
    iterations = [i+1 for i in range(num_iterations)]
    lrs = [scheduler(i) for i in iterations]
    display.set_matplotlib_formats('svg')
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()
```

### Stepwise Decay Schedule

We already know `FactorScheduler`, let's plot how
it decays the learning rate.

```{.python .input}
plot(scheduler)
```

We can define non-uniform intervals with `MultiFactorScheduler` and in the
example below we halve the learning rate _after_ the 250th, 750th (i.e. a step
length of 500 iterations) and 900th (a step length of 150 iterations). As
before, the learning rate of the 250th iteration will be 1 and the 251th
iteration will be 0.5.

```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(base_lr=1, step=[250, 750, 900], factor=0.5)
plot(scheduler)
```

### Polynomial Schedule

`PolyScheduler` gives a smooth decay using a polynomial
function and reaches a learning rate of 0 after `max_update` iterations. In the
example below, we have a quadratic function (`pwr=2`) that falls from 1 to 0.001
at iteration 800. After this the learning rate stays at 0.001.

```{.python .input}
scheduler = lr_scheduler.PolyScheduler(base_lr=1, final_lr=1e-3, max_update=800, pwr=2)
plot(scheduler)
```

## Cosine Schedules

`CosineScheduler` decays the learning rate by using the
cosine function. It is also a smooth decay but no needs to choose the function
type compared to `PolyScheduler`.

```{.python .input}
plot(lr_scheduler.CosineScheduler(base_lr=1, final_lr=1e-3, max_update=800))
```

## Warming Up

Sometimes the initial learning rate is too large to converge. We
may perform an additional warming up step to increase the learning rate from a
small value. For example, we start with value 0, and linearly increase it to the
initial learning rate for the first 100 iterations.

```{.python .input}
# The warming up applies to other scheduler as well. 
plot(lr_scheduler.CosineScheduler(base_lr=1, final_lr=1e-3, max_update=800, 
                                  warmup_steps=100, warmup_begin_lr=0, warmup_mode='linear'))
```

## Customized Schedulers

We can implement our own custom schedule with a
function or callable class, that takes an integer denoting the iteration index
(starting at 1) and returns a float representing the learning rate to be used
for that iteration. We implement the cosine schedule in the example below as a
callable class (see `__call__` method).

```{.python .input}
class CosineScheduler():
    def __init__(self, base_lr, final_lr, max_update):
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.max_update = max_update
    def __call__(self, iteration):
        if iteration <= self.max_update:
            unit = (1 + math.cos(iteration * math.pi / self.max_update)) / 2
            return (unit * (self.base_lr - self.final_lr)) + self.final_lr
        else:
            return self.final_lr
plot(CosineScheduler(base_lr=1, final_lr=1e-3, max_update=800))
```
