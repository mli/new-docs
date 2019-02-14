# Initialization

<!-- adapted from diveintodeeplearning -->

In the previous examples we played fast and loose with setting up our networks.
In particular we did the following things that *shouldn't* work:

* We defined the network architecture with no regard to the input
  dimensionality.
* We added layers without regard to the output dimension of the previous layer.
* We even 'initialized' these parameters without knowing how many parameters
  were were to initialize.

All of those things sound impossible and indeed, they are. After
all, there's no way MXNet (or any other framework for that matter) could predict
what the input dimensionality of a network would be. Later on, when working with
convolutional networks and images this problem will become even more pertinent,
since the input dimensionality (i.e. the resolution of an image) will affect the
dimensionality of subsequent layers at a long range. Hence, the ability to set
parameters without the need to know at the time of writing the code what the
dimensionality is can greatly simplify statistical modeling. In what follows,
we will discuss how this works using initialization as an example. After all,
we cannot initialize variables that we don't know exist.

In what follows, we will discuss how this works using initialization as an
example.

## Instantiating a Network

Let's see what happens when we instantiate a network. We start with our
trusty multilayer perceptron as before.

```{.python .input}
from mxnet import init, nd
from mxnet.gluon import nn

def getnet():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = getnet()
```

At this point the network doesn't really know yet what the dimensionalities of
the various parameters should be. All one could tell at this point is that each
layer needs weights and bias, albeit of unspecified dimensionality. If we try
accessing the parameters, that's exactly what happens.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

You'll notice `None` here in each `Dense` layer. This absense of value is how
MXNet keeps track of unspecified dimensionality. In particular, trying to access
`net[0].weight.data()` at this point would trigger a runtime error stating that
the network needs initializing before it can do anything. 

Note that if we did want to specify dimensionality, we could have done so by
using the kwarg `in_units`, e.g. `Dense(256, activiation='relu', in_units=20)`.

Let's see whether anything changes after we initialize the parameters:


```{.python .input}
net.initialize()
net.collect_params()
```

As we can see, nothing really changed. Only once we provide the network with
some data do we see a difference. Let's try it out.

```{.python .input}
x = nd.random.uniform(shape=(2, 20))
net(x)            # Forward computation.

net.collect_params()
```

We see all the dimensions have been determined and the parameters initialized.
This is because dimensional inference and parameter initialization have been
performed in a lazy manner, so they are performed only when needed. In the
above case, they are performed as a prerequisite to the forward computation.

Dimensional inference works like this: as soon as we knew the input
dimensionality, $\mathbf{x} \in \mathbb{R}^{20}$ it was possible to define the
weight matrix for the first layer, i.e. $\mathbf{W}_1 \in \mathbb{R}^{256 \times
20}$. With that out of the way, we can progress to the second layer, define its
dimensionality to be $10 \times 256$ and so on through the computational graph
and bind all the dimensions as they become available. 

Once this is known, we can proceed by initializing parameters. This is the
solution to the three problems outlined above.


## Deferred Initialization in Practice

Now that we know how it works in theory, let's see when the initialization is
actually triggered. In order to do so, we mock up an initializer which does
nothing but report a debug message stating when it was invoked and with which
paramers.

```{.python .input  n=22}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # The actual initialization logic is omitted here.

net = getnet()
net.initialize(init=MyInit())
```

Note that, although `MyInit` will print information about the model parameters
when it is called, the above `initialize` function does not print any
information after it has been executed.  Therefore there is no real
initialization parameter when calling the `initialize` function. Next, we define
the input and perform a forward calculation.

```{.python .input  n=25}
x = nd.random.uniform(shape=(2, 20))
y = net(x)
```

At this time, information on the model parameters is printed. When performing a
forward calculation based on the input `x`, the system can automatically infer
the shape of the weight parameters of all layers based on the shape of the
input. Once the system has created these parameters, it calls the `MyInit`
instance to initialize them before proceeding to the forward calculation.

Of course, this initialization will only be called when completing the initial
forward calculation. After that, we will not re-initialize when we run the
forward calculation `net(x)`, so the output of the `MyInit` instance will not be
generated again.

```{.python .input}
y = net(x)
```

As mentioned at the beginning of this section, deferred initialization can also
cause confusion. Before the first forward calculation, we were unable to
directly manipulate the model parameters, for example, we could not use the
`data` and `set_data` functions to get and modify the parameters. Therefore, we
often force initialization by sending a sample observation through the network.

## Forced Initialization

Deferred initialization does not occur if the system knows the shape of all
parameters when calling the `initialize` function. This can occur in two cases:

* We've already seen some data and we just want to reset the parameters.
* We specificed all input and output dimensions of the network when defining it.

The first case works just fine, as illustrated below.

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

The second case requires us to specify the remaining set of parameters when
creating the layer. For instance, for dense layers we also need to specify the
`in_units` so that initialization can occur immediately once `initialize` is
called.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```

## Parameter Initialization

Now that we know how to access the parameters, let's look at how to initialize
them properly. We discussed the need for
[Initialization](../chapter_deep-learning-basics/numerical-stability-and-init.md)
in the previous chapter. By default, MXNet initializes the weight matrices
uniformly by drawing from $U[-0.07, 0.07]$ and the bias parameters are all set
to $0$. However, we often need to use other methods to initialize the weights.
MXNet's `init` module provides a variety of preset initialization methods, but
if we want something out of the ordinary, we need a bit of extra work.

### Built-in Initialization

Let's begin with the built-in initializers. The code below initializes all
parameters with Gaussian random variables.

```{.python .input  n=9}
# force_reinit ensures that the variables are initialized again, regardless of whether they were
# already initialized previously.
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

If we wanted to initialize all parameters to 1, we could do this simply by
changing the initializer to `Constant(1)`.

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

If we want to initialize only a specific parameter in a different manner, we
can simply set the initializer only for the appropriate subblock (or parameter)
for that matter. For instance, below we initialize the second layer to a
constant value of 42 and we use the `Xavier` initializer for the weights of the
first layer.

```{.python .input  n=11}
net[1].initialize(init=init.Constant(42), force_reinit=True)
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[1].weight.data()[0,0])
print(net[0].weight.data()[0])
```

### Custom Initialization

Sometimes, the initialization methods we need are not provided in the `init`
module. At this point, we can implement a subclass of the `Initializer` class
so that we can use it like any other initialization method. Usually, we only
need to implement the `_init_weight` function and modify the incoming NDArray
according to the initial result. In the example below, we  pick a decidedly
bizarre and nontrivial distribution, just to prove the point. We draw the
coefficients from the following distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U[-10, -5] & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

If even this functionality is insufficient, we can set parameters directly.
Since `data()` returns an NDArray we can access it just like any other matrix.
A note for advanced users - if you want to adjust parameters within an
`autograd` scope you need to use `set_data` to avoid confusing the automatic
differentiation mechanics.

```{.python .input  n=13}
net[0].weight.data()[:] += 1
net[0].weight.data()[0,0] = 42
net[0].weight.data()[0]
```

## Tied Parameters

In some cases, we want to share model parameters across multiple layers. For
instance when we want to find good word embeddings we may decide to use the
same parameters both for encoding and decoding of words. We discussed one such
case when we introduced [Blocks](model-construction.md). Let's see how to do
this a bit more elegantly. In the following we allocate a dense layer and then
use its parameters specifically to set those of another layer.

```{.python .input  n=14}
net = nn.Sequential()
# we need to give the shared layer a name such that we can reference its parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0,0] = 100
# And make sure that they're actually the same object rather than just having the same value.
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

The above exampe shows that the parameters of the second and third layer are
tied. They are identical rather than just being equal. That is, by changing one
of the parameters the other one changes, too. What happens to the gradients is
quite ingenious. Since the model parameters contain gradients, the gradients of
the second hidden layer and the third hidden layer are accumulated in the
`shared.params.grad( )` during backpropagation.

