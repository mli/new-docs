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

L1Loss also called mean abolsute error computes the sum of absolute distance between target values and the output of the neural network. It is defined as:

$$ L = \sum_i \vert {label}_i - {pred}_i \vert. $$

It is a non-smooth function that can lead to non-convergence. It creates the same gradient for small and large loss values, which can be problematic for the learning process.

```{.python .input}
show_regression_loss(gloss.L1Loss())
```

#### [L2 Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss)

L2Loss also called Meaned Squared Error is a regression loss function that computes the squared distances between the target values and the output of the neural network. It is defined as:

$$ L = \frac{1}{2} \sum_i \vert {label}_i - {pred}_i \vert^2. $$

Compared to L1, L2 loss it is a smooth function and it creates larger gradients for large loss values. However due to the squaring it puts high weight on outliers. 

```{.python .input}
show_regression_loss(gloss.L2Loss())
```

#### [Huber Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HuberLosss)

HuberLoss  combines advantages of L1 and L2 loss. It calculates a smoothed L1 loss that is equal to L1 if the adsolute error exceeds a threshold $$\rho$$, otherwise it is equal to L2. It is defined as:
$$ 
\begin{split}L = \sum_i \begin{cases} \frac{1}{2 {rho}} ({label}_i - {pred}_i)^2 &
                   \text{ if } |{label}_i - {pred}_i| < {rho} \\
                   |{label}_i - {pred}_i| - \frac{{rho}}{2} &
                   \text{ otherwise }
    \end{cases}\end{split}
$$

```{.python .input}
show_regression_loss(gloss.HuberLoss(rho=1))
```

#### [Cross Entropy Loss with Sigmoid](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss)

Binary Cross Entropy is a loss function used for binary classification problems e.g. classifying images into 2 classes. Cross entropy measures the difference between two propbaility distributions and it is defined as:
$$\sum_i -{(y\log(p) + (1 - y)\log(1 - p))} $$
Before the loss is computed a sigmoid activation is applied per default. You can avoid that by setting ```from_sigmoid``` to False. 

```{.python .input}
show_classification_loss(gloss.SigmoidBinaryCrossEntropyLoss())
```

#### [Cross Entropy Loss with Softmax](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss)

In classification, we often apply the
softmax operator to the predicted outputs to obtain prediction probabilities,
and then apply the cross entropy loss against the true labels: 

$$ \begin{align}\begin{aligned}p = \softmax({pred})\\L = -\sum_i \sum_j {label}_j \log p_{ij}\end{aligned}\end{align}
$$

Running these two
steps one-by-one, however, may lead to numerical instabilities. The `loss`
module provides a single operators with softmax and cross entropy fused to avoid
such problem.

```{.python .input}
loss = gloss.SoftmaxCrossEntropyLoss()
x = nd.array([[1, 10], [8, 2]])
y = nd.array([0, 1])
loss(x, y)
```

#### [Hinge Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss)

Classifcation problems normally require a zero-one loss function, which assigns 0 loss to correct classifications and 1 otherwise. The problem of such a function is that it is hard to optimize and its gradients would be zero. Hinge Loss creates an upper bound on the zero-one loss function which makes it a convex and continous function.  It is defined as:

$$
L = \sum_i max(0, {margin} - {pred}_i \cdot {label}_i)
$$

```{.python .input}
show_classification_loss(gloss.HingeLoss())
```

#### [Logistic Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.LogisticLoss)

The Logistic Loss function computes the performance of binary classification models. 
$$
L = \sum_i \log(1 + \exp(- {pred}_i \cdot {label}_i))
$$
The log loss decreases the closer the prediction is to the actual label. It is sensitive to outliers, because incorrectly classified points are penalized more.

```python
show_classification_loss(gloss.LogisticLoss())
```

#### [Kullback-Leibler Divergence Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.KLDivLoss)

The Kullback-Leibler divergence loss measures the divergence between two propbaility distributions by calculating the difference between cross entropy and entropy. It takes as input the probaility of true label and the probabilty of predicted label. 

$$
L = \sum_i {label}_i * \big[\log({label}_i) - {pred}_i\big]
$$

The loss is large, if the prediced probality distribution is far from the groundtruth propability distribution. KL divergence is an asymmetric measure. KL divergence loss can be used in Variational Autoencoders (VAEs), and reinforcement learning policy networks such as Trust Region Policy Optimization (TRPO)


For instance in the following example we get a KL divergence of 0.02. We set ```from_logits=False```, so the loss functions will apply ```log_softmax``` on the network output, before computing the KL divergence.

```python
output = mx.nd.array([[0.39056206, 1.3068528, 0.39056206, -0.30258512]])
print('output.softmax(): {}'.format(output.softmax().asnumpy().tolist()))
target_dist = mx.nd.array([[0.3, 0.4, 0.1, 0.2]])
loss_fn = mx.gluon.loss.KLDivLoss(from_logits=False)
loss = loss_fn(output, target_dist)
print('loss (kl divergence): {}'.format(loss.asnumpy().tolist()))
```

#### [Triplet Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.TripletLoss)

Triplet loss takes three input tensors and measures the relative similarity. It takes a positive and negative input and the anchor. 

$$
L = \sum_i \max(\Vert {pos_i}_i - {pred} \Vert_2^2 -
    \Vert {neg_i}_i - {pred} \Vert_2^2 + {margin}, 0)
$$

The loss function minimizes the distance between similar inputs and maximizes the distance  between dissimilar ones.  
In the case of learning embeddings for images of characters, the network may get as input the following 3 images: 

![triplet_loss](triplet_loss.png)

The network would learn to minimize the distance between the two `A`'s and maximize the distance between `A` and `Z`.

#### [CTC Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.CTCLoss)

CTC Loss is the connectionist temporal classification loss. It is used to train recurrent neural networks with variable time dimension. It learns the alignment and labelling of input sequences. It takes a sequence as input and gives propabilies for each timestep. For instance in the followng image the word is not well algined with the 5 timesteps because of the different sizes of characters. CTC Loss finds for each timestep the highest propability e.g. `t1` presents with high propability a `C`. It combines the highest probapilities and returns the best path decoding. For an in-depth tutorial on how to use CTC-Loss in MXNet, check out this [example](https://github.com/apache/incubator-mxnet/tree/master/example/ctc).

![ctc_loss](ctc_loss.png)

#### [Cosine Embedding Loss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.CosineEmbeddingLoss)
The cosine embedding loss computes the cosine distance between two input vectors. 

$$
\begin{split}L = \sum_i \begin{cases} 1 - {cos\_sim({input1}_i, {input2}_i)} & \text{ if } {label}_i = 1\\
                 {cos\_sim({input1}_i, {input2}_i)} & \text{ if } {label}_i = -1 \end{cases}\\
cos\_sim(input1, input2) = \frac{{input1}_i.{input2}_i}{||{input1}_i||.||{input2}_i||}\end{split}
$$

Cosine distance measures the similarity between two tensors given a label and is typically used for learning nonlinear embeddings.  
For instance in the following code example we measure the similarity between the input vectors `x` and `y`. Since they are the same the label equals `1`. The loss function returns $$ \sum_i 1 - {cos\_sim({input1}_i, {input2}_i)} $$ which is equal `0`.
```python
x = mx.nd.array([1,0,1,0,1,0])
y = mx.nd.array([1,0,1,0,1,0])
label = mx.nd.array(1)
loss = gluon.loss.CosineEmbeddingLoss()
print(loss(x,y,label))
```
Now let's make `y` the opposite of `x`, so we set the label `-1` and the function will return  $$ \sum_i cos\_sim(input1, input2) $$

```python
x = mx.nd.array([1,0,1,0,1,0])
y = mx.nd.array([0,1,0,1,0,1])
label = mx.nd.array(-1)
loss = gluon.loss.CosineEmbeddingLoss()
print(loss(x,y,label))
```


#### [PoissonNLLLoss](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/loss.html#mxnet.gluon.loss.PoissonNLLLoss)
Poisson distribution is widely used for modelling count data. It is defined as:

$$
f(x) = \frac{\mu ^ {\kern 0.08 em x} e ^ {-\mu}} {x!} \qquad \qquad x = 0,1,2 , \ldots \,.
$$


For instance the count of cars in road traffic approximatly follows a Poisson distribution. Using an ordinary least squares model for Poisson distributed data would not work well because of two reasons: 
 - count data cannot be negative 
 - variance may not be constant

Instead we can use a Poisson regression model, also known as log-linear model. Thereby the Poisson incident rate $$\mu$$ is 
modelled by a linear combination of unknown parameters.
We can then use the PoissonNLLLoss which calculates the negative log likelihood for a target that follows a Poisson distribution. 

$$ L = \text{pred} - \text{target} * \log(\text{pred}) +\log(\text{target!}) $$



