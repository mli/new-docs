# Loss functions

A loss function measures the difference between predicted
outputs and the
according label. We can then run back propogation to compute the
gradients. Let's first import the modules, where the `mxnet.gluon.loss` module
is imported as `gloss` to avoid the commonly used name `loss`.

```{.python .input}
from IPython import display
from matplotlib import pyplot as plt
from mxnet import nd, autograd
from mxnet.gluon import nn, loss as gloss  
```



## Basic Usages

Now let's create an instance of the $\ell_2$ loss.

```{.python .input}
loss = gloss.L2Loss()
```

and then feed two inputs to compute the elemental-wise loss values.

```{.python .input}
x = nd.ones((2,))
y = nd.ones((2,)) * 2
loss(x, y)
```

These values should be equal to the math definition: $0.5\|x-y\|^2$.

```{.python .input}
.5 * (x - y)**2
```

In a mini-batch, some examples may be more important than others. We can apply
weights to individual examples during the forward function (the default weight
value is 1).

```{.python .input}
loss(x, y, nd.array([1, 2]))
```

Next we show how to use a loss function to compute gradients.

```{.python .input}
X = nd.random.uniform(shape=(2, 4)) 
net = nn.Dense(1)
net.initialize()
with autograd.record():
    l =  loss(net(X), y)
l
```

Since the both network forward and loss are recorded, we can compute the
gradients w.r.t. the loss function.

```{.python .input}
l.backward()
net.weight.grad()
```

## Loss functions

Most commonly used loss functions can be divided into 2 classes: regression and classification. Regression loss functions output real-values, while classification loss functios output a class.

Let's first visualize several regression losses. We
visualize the loss values versus the predicted values with label values fixed to
be 0.

```{.python .input}
def plot(x, y):
    display.set_matplotlib_formats('svg')
    plt.plot(x.asnumpy(), y.asnumpy())
    plt.xlabel('x')
    plt.ylabel('loss')
    plt.show()
    
def show_regression_loss(loss):
    x = nd.arange(-5, 5, .1)
    y = loss(x, nd.zeros_like(x))
    plot(x, y)  

```

Then plot the classification losses with label values fixed to be 1.

```python
def show_classification_loss(loss):
		x = nd.arange(-5, 5, .1)
    y = loss(x, nd.ones_like(x))
    plot(x, y) 
```

#### [L1 Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss)

L1Loss also called mean abolsute error computes the sum of absolute distance between target values and the output of the neural network. It is a non-smooth function that can lead to non-convergence. It creates the same gradient for small and large loss values, which can be problematic for the learning process.

```{.python .input}
show_regression_loss(gloss.L1Loss())
```

#### [L2 Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss)

L2Loss also called Meaned Squared Error is a regression loss function that computes the squared distances between the target values and the output of the neural network. Compared to L1, L2 loss it is a smooth function and it creates larger gradients for large loss values. However due to the squaring it puts high weight on outliers. 

```{.python .input}
show_regression_loss(gloss.L2Loss())
```

####[Huber Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HuberLosss)

HuberLoss  combines advantages of L1 and L2 loss. It calculates a smoothed L1 loss that is equal to L1 if the adbsolute error exceeds a threshold rho, otherwise it is equal to L2.

```{.python .input}
show_regression_loss(gloss.HuberLoss(rho=1))
```

####[Cross Entropy Loss with Sigmoid](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss)

Binary Cross Entropy is a loss function used for binary classification problems e.g. classifying images into 2 classes.  Cross entropy measures the difference between two propbaility distributions. Before the loss is computed a sigmoid activation is applied. 

```{.python .input}
show_classification_loss(gloss.SigmoidBinaryCrossEntropyLoss())
```

####[Cross Entropy Loss with Softmax](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss)

In classification, we often apply the
softmax operator to the predicted outputs to obtain prediction probabilities,
and then apply the cross entropy loss against the true labels. Running these two
steps one-by-one, however, may need to numerical instabilities. The `loss`
module provides a single operators with softmax and cross entropy fused to avoid
such problem.

```{.python .input}
loss = gloss.SoftmaxCrossEntropyLoss()
x = nd.array([[1, 10], [8, 2]])
y = nd.array([0, 1])
loss(x, y)
```

#### [Hinge Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss)

Classifcation problems normally require a zero-one loss function, which assigns 0 loss to correct classifications and 1 otherwise. The problem of such a function is that it is hard to optimize and its gradients would be zero. Hinge Loss creates an upper bound on the zero-one loss function which makes it a convex and continous function. 

```{.python .input}
show_classification_loss(gloss.HingeLoss())
```

#### [Logistic Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.LogisticLoss)

The Logistic Loss function computes the performance of binary classification models. The log loss decreases the closer the prediction is to the actual label. It is sensitive to outliers, because incorrectly classified points are penalized more.

```python
show_classification_loss(gloss.LogisticLoss())
```



#### [Kullback-Leibler Divergence Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.KLDivLoss)

The Kullback-Leibler divergence loss measures the divergence between two propbaility distributions by caclucating the cross entropy minus the entropy. 

```python
gloss.KLDivLoss()
```



#### [Triplet Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.TripletLoss)

Triplet loss takes three input tensors and measures the relative similarity. It takes a positive and negative input tensor and the predicted tensos. The loss function minimizes the distance between similar inputs and maximizes the distance  between dissimilar inputs.

```
show_classification_loss(gloss.TripletLoss())
```



####[CTC Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.CTCLoss)

CTC Loss is the connectionist temporal classification loss. It is used to train recurrent neural networks with variable time dimension. It learns the alignment and labelling of input sequences. 