# Trainer

A trainer updates neural network parameters by an optimization
method.

## Basic Usages

We first create a simple Perceptron and run forward and
backward passes to obtain the gradients.

```{.python .input}
from mxnet import gluon, nd, autograd, optimizer

loss = gluon.loss.L2Loss()
net = gluon.nn.Dense(1)
net.initialize()
batch_size = 8
X = nd.random.uniform(shape=(batch_size, 4))
y = nd.random.uniform(shape=(batch_size,))

def forward_backward():
    with autograd.record():
        l = loss(net(X), y)
    l.backward()
forward_backward()
```

Next create a `Trainer` instance with specifying the network parameters and
optimization method, which is plain SGD with learning rate $\eta=1$.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(),
                        optimizer='sgd', optimizer_params={'learning_rate':1})
```

Before updating, let's check the current weight.

```{.python .input}
cur_weight = net.weight.data().copy()
```

Now call the `step` method to perform one update. It accepts the `batch_size` as
an argument to normalize the gradients. We can see the weight is changed.

```{.python .input}
trainer.step(batch_size)
net.weight.data()
```

Since we used plain SGD, so the updating rule is $w = w - \eta/b \nabla \ell$,
where $b$ is the batch size and $\ell$ is the loss function. We can verify it:

```{.python .input}
cur_weight - net.weight.grad() * 1 / batch_size
```

## Use another Optimization Method

In the previous example, we use argument
`optimizer` to select the optimization method, and `optimizer_params` to specify
the optimization method arguments. All pre-defined optimization methods are
provided in the `mxnet.optimizer` module.

We can pass an optimizer instance directly to the trainer. For example:

```{.python .input}
optim = optimizer.Adam(learning_rate = 1)
trainer = gluon.Trainer(net.collect_params(), optim)

```

```{.python .input}
forward_backward()
trainer.step(batch_size)
net.weight.data()
```

For all implemented methods, please refer to the
[API reference](http://beta.mxnet.io/api/gluon-related/mxnet.optimizer.html) for
the `optimizer` module. Besides, the
[Dive into Deep Learning](http://en.diveintodeeplearning.org/chapter_optimization/index.html)
book explains each optimization methods from scratch.

## Change Learning Rate
We set the initial learning rate when creating an trainer. But we often need to
change the learning rate during training. The current training rate can be
accessed through the `learning_rate` attribute.

```{.python .input}
trainer.learning_rate
```

We can change it through the `set_learning_rate` method.

```{.python .input}
trainer.set_learning_rate(0.1)
trainer.learning_rate
```

In addition, multiple pre-defined learning rate scheduling methods are
implemented in the `mxnet.lr_scheduler` module. Please refer to [its
tutorial](../lr_scheduler.md).
