# Normalization Blocks

When training deep neural networks there are a number of techniques that are thought to be essential for model convergence. One important area is deciding how to initialize the parameters of the network. Using techniques such as Xavier initialization, we can can improve the gradient flow through the network at the start of training. Another important technique is normalization: i.e. scaling and shifting certain values towards a distribution with a mean of 0 (i.e. zero centered) and a standard distribution of 1 (i.e. unit variance). Which values you normalize depends on the exact method used as we'll see later on.

<p align="center">
    <img src="./imgs/data_normalization.jpeg" alt="drawing" width="500"/>
    <p align="center">Figure 1: Data Normalization
        <a href="http://cs231n.github.io/neural-networks-2/">(Source)</a>
    </p>
</p>

Why does this help? [Some research](https://papers.nips.cc/paper/7515-how-does-batch-normalization-help-optimization.pdf) has found that networks with normalization have a loss function that's easier to optimize using stochastic gradient descent. Other reasons are that it prevents saturation of activations and prevents certain features from dominating due to differences in scale.

### Data Normalization

One of the first applications of normalization is on the input data to the network. Although this whole section can be skipped in favor of an initial `BatchNorm` layer in your network, which will have a very similar effect (see next section).

* **Step 1** is to calculate the mean and standard deviation of the entire training dataset. You'll usually want to do this for each channel separately. Sometimes you'll see normalization on images applied per pixel, but per channel is more common.
* **Step 2** is to use these statistics to normalize each batch for training and for inference too.

Warning: You should calculate the normalization means and standard deviations using the training dataset only. Any leakage of information from you testing dataset will effect the reliability of your testing metrics.

When using pre-trained models from the Gluon Model Zoo you'll sometimes see the normalization statistics used for training. You'll want to use these statistics to normalize your own input data for fine-tuning or inference with these models. Using `transforms.Normalize` is one way of applying the normalization, and this should be used in the `Dataset`.


```python
import mxnet as mx
from mxnet.gluon.data.vision.transforms import Normalize

image_int = mx.nd.random.randint(low=0, high=256, shape=(1,3,2,2))
image_float = image_int.astype('float32')/255
normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image = normalizer(image_float)
image
```




    
    [[[[-1.31304061  1.17004883]
       [ 2.18040919 -0.35405433]]
    
      [[ 1.65826333  2.20098042]
       [-0.51260501  1.74579835]]
    
      [[ 0.63564289  1.55939019]
       [-1.12470579  1.22823548]]]]
    <NDArray 1x3x2x2 @cpu(0)>



### Activation Normalization

We don't have to limit ourselves to normalizing the inputs to the network either. A similar idea can be applied inside the network too, and we can normalize activations between certain layer operations. With deep neural networks most of the convergence benefits described are from this type of normalization.

MXNet Gluon has 3 of the most commonly used normalization blocks: `BatchNorm`, `LayerNorm` and `InstanceNorm`. You can use them in networks just like any other MXNet Gluon Block, and are often used after `Activation` Blocks.

Watch Out: Check the architecture of models carefully because sometimes the normalization is applied before the `Activation`.

Advanced: all of the following methods begin by normalizing certain input distribution (i.e. zero centered with unit variance), but then shift by (a trainable parameter) beta and scale by (a trainable parameter) gamma. Overall the effect is changing the input distribution to have a mean of beta and a variance of gamma, also allowing to the network to 'undo' the effect of the normalization if necessary.

## Batch Normalization

Figure 1: `BatchNorm` on NCHW data | Figure 2: `BatchNorm` on NTC data 
- | - 
![alt](./imgs/NCHW_BN.png) | ![alt](./imgs/NTC_BN.png)
(e.g. batch of images) using the default of `axis=1` | (e.g. batch of sequences) overriding the default with `axis=2` (or `axis=-1`)

One of the most popular normalization techniques is batch normalization (often called BatchNorm). We normalize the activations **across all samples in a batch** for each of the channels independently. See Figure 1. We calculate two batch (or local) statistics for every channel to perform the normalization: the mean and variance of the activations in that channel for all samples in a batch. And we use these to shift and scale respectively.

Tip: we can used this at the start of a network to perform data normalization, although this is not exactly equivalent to the example seen above.

Warning: the estimates for the batch mean and variance can themselves have high variance when the batch size is small (or when the spatial dimensions of samples are small). This can lead to instability during training, and unreliable estimates for the global statistics.

Warning: it seems that `BatchNorm` is better suited to convolutional networks (CNNs) than recurrent networks (RNNs). We expect the input distribution to the recurrent cell to change over time, so normalization over time doesn't work well. `LayerNorm` is better suited for this case. When you do *need* to use `BatchNorm` on sequential data, make sure the `axis` parameter is set correctly. With data in NTC format you should set `axis=2` (or `axis=-1` equivalently). See Figure 2.

As an example, we'll apply `BatchNorm` to a batch of 2 samples, each with 2 channels, and both height and width of 2 (in NCHW format).


```python
data = mx.nd.arange(start=0, stop=2*2*2*2).reshape(2, 2, 2, 2)
print(data)
```

    
    [[[[  0.   1.]
       [  2.   3.]]
    
      [[  4.   5.]
       [  6.   7.]]]
    
    
     [[[  8.   9.]
       [ 10.  11.]]
    
      [[ 12.  13.]
       [ 14.  15.]]]]
    <NDArray 2x2x2x2 @cpu(0)>


With MXNet Gluon we can apply batch normalization with the `mx.gluon.nn.BatchNorm` block. It can be created and used just like any other MXNet Gluon block (such as `Conv2D`). Its input will typically be unnormalized activations from the previous layer, and the output will be the normalized activations ready for the next layer. Since we're using data in NCHW format we can use the default axis.


```python
net = mx.gluon.nn.BatchNorm()
```

We still need to initialize the block because it has a number of trainable parameters, as we'll see later on.


```python
net.initialize()
```

We can now run the network as we would during training (under `autograd.record` context scope).

Remember: `BatchNorm` runs differently during training and inference. When training, the batch statistics are used for normalization. During inference, a exponentially smoothed average of the batch statistics that have been observed during training is used instead.

Warning: `BatchNorm` assumes the channel dimension is the 2nd in order (i.e. `axis=1`). You need to ensure your data has a channel dimension, and change the `axis` parameter of `BatchNorm` if it's not the 2nd dimension. A batch of greyscale images of shape `(100,32,32)` would not work, since the 2nd dimension is height and not channel. You'd need to add a channel dimension using `data.expand_dims(1)` in this case to give shape `(100,1,32,32)`.


```python
with mx.autograd.record():
    output = net(data)
    loss = output.abs()
loss.backward()
print(output)
```

    
    [[[[-1.32424402 -1.08347237]
       [-0.84270072 -0.60192907]]
    
      [[-1.32424402 -1.08347237]
       [-0.84270072 -0.60192907]]]
    
    
     [[[ 0.60192907  0.84270072]
       [ 1.08347237  1.32424402]]
    
      [[ 0.60192907  0.84270072]
       [ 1.08347237  1.32424402]]]]
    <NDArray 2x2x2x2 @cpu(0)>


We can immediately see the activations have been scaled down and centered around zero. Activations are the same for each channel, because each channel was normalized independently. We can do a quick sanity check on these results, by manually calculating the batch mean and variance for each channel.


```python
batch_means = data.mean(axis=1, exclude=True)
batch_vars = (data - batch_means.reshape(1, -1, 1, 1)).square().mean(axis=1, exclude=True)
print('batch_means:', batch_means.asnumpy())
print('batch_vars:', batch_vars.asnumpy())
```

    batch_means: [ 5.5  9.5]
    batch_vars: [ 17.25  17.25]


And use these to scale the first entry in `data`, to confirm the `BatchNorm` calculation of `-1.324` was correct.


```python
print("manually calculated:", ((data[0][0][0][0] - batch_means[0])/batch_vars[0].sqrt()).asnumpy())
print("automatically calculated:", output[0][0][0][0].asnumpy())
```

    manually calculated: [-1.3242445]
    automatically calculated: [-1.32424402]


As mentioned before, `BatchNorm` has a number of parameters that update throughout training. 2 of the parameters are not updated in the typical fashion (using gradients), but instead are updated deterministically using exponential smoothing. We need to keep track of the average mean and variance of batches during training, so that we can use these values for normalization during inference.

Advanced: Often during inference, we have a batch size of 1 so batch variance would be impossible to calculate. And we might get a data distribution shift between training and inference data, which shouldn't just be normalized away.

Warning: `use_global_stats=True` is a method of using the global running statistics during training, but this will prevent the global statistics from updating. Also, it has no effect on inference mode. Only use this for pre-trained models where the global statistics have been calculated already and you want to use them during training. One example would be using a pre-trained ResNet as a image feature extractor in an Instance Segmentation model that can only be trained with small batches.

After a single step (specifically after the `backward` call) we can see the `running_mean` and `running_var` have been updated.


```python
print('running_mean:', net.running_mean.data().asnumpy())
print('running_var:', net.running_var.data().asnumpy())
```

    running_mean: [ 0.55000013  0.95000023]
    running_var: [ 2.62500048  2.62500048]


You should notice though that these running statistics do not match the batch statistics we just calculated. And instead they are just 10% of the value we'd expect. We see this because of the exponential average process, and because the `momentum` parameter of `BatchNorm` is equal to 0.9 : i.e. 10% of the new value, 90% of the old value (which was initialized to 0). Over time the running statistics will converge to the statistics of the input distribution, while still being flexible enough to adjust to shifts in the input distribution. Using the same batch another 100 times (which wouldn't happen in practice), we can see the running statistics converge to the batch statsitics calculated before.


```python
for i in range(100):
    with mx.autograd.record():
        output = net(data)
        loss = output.abs()
    loss.backward()
print('running_means:', net.running_mean.data().asnumpy())
print('running_vars:', net.running_var.data().asnumpy())
```

    running_means: [ 5.49986982  9.49976826]
    running_vars: [ 17.24960899  17.24960899]


#### Beta and Gamma

As mentioned previously, there are two additional parameters in `BatchNorm` which are trainable in the typical fashion (with gradients). `beta` is used to shift and `gamma` is used to scale the normalized distribution, which allows the network to 'undo' the effects of normalization if required. You can prevent `beta` shifting by setting `center=False`, and prevent `gamma` scaling by setting `scale=False` when initializing the `BatchNorm` block.

Watch out: zero centering and scaling to unit variance will still occur when setting `center=False` and `scale=False`. Only post normalization shifting and scaling will prevented.

Advanced: although the `beta` and `gamma` parameters will still exist, they won't update (because their `grad_req=null`).

We haven't updated these parameters yet, so they should still be as initialized. You can see the default for `beta` is 0 (i.e. not shift) and `gamma` is 1 (i.e. not scale), so the initial behaviour is to keep the distribution unit normalized.


```python
print('beta:', net.beta.data().asnumpy())
print('gamma:', net.gamma.data().asnumpy())
```

    beta: [ 0.  0.]
    gamma: [ 1.  1.]


We can also check the gradient on these parameters. Since we were finding the gradient of the sum of absolute values, we would expect the gradient of `gamma` to be equal to the number of points in the data (i.e. 16). So to minimize the loss we'd decrease the value of `gamma`, which would happen as part of a `trainer.step`.


```python
print('beta gradient:', net.beta.grad().asnumpy())
print('gamma gradient:', net.gamma.grad().asnumpy())
```

    beta gradient: [ 0.  0.]
    gamma gradient: [ 7.70469236  7.70469236]


#### Inference Mode

When it comes to inference, `BatchNorm` uses the global statistics that were calculated during training. Since we're using the same batch of data over and over again (and our global running statistics have converged), we get a very similar result to using training mode. `beta` and `gamma` are also applied by default (unless explicitly removed).


```python
output = net(data)
print(output)
```

    
    [[[[-1.32422769 -1.08345342]
       [-0.84267896 -0.60190463]]
    
      [[-1.32420325 -1.08342886]
       [-0.84265453 -0.60188013]]]
    
    
     [[[ 0.60196728  0.84274167]
       [ 1.083516    1.32429039]]
    
      [[ 0.60199177  0.84276611]
       [ 1.08354056  1.32431483]]]]
    <NDArray 2x2x2x2 @cpu(0)>


# Layer Normalization

An alternative to `BatchNorm` that is better suited to recurrent networks (RNNs) is called `LayerNorm`. Unlike `BatchNorm` which normalizes across all samples of a batch per channel, `LayerNorm` normalizes **across all channels of a single sample**.

Some of the disadvantages of `BatchNorm` no longer apply. Small batch sizes are no longer an issue, since normalization statistics are calculated on single samples. And confusion around training and inference modes disappears because `LayerNorm` is the same for both modes.

Warning: similar to having a small batch sizes in `BatchNorm`, you may have issues with `LayerNorm` if the input channel size is small. Using embeddings with a large enough dimension size avoids this (approx >20).

Warning: currently MXNet Gluon's implementation of `LayerNorm` is applied along a single axis (which should be the channel axis). Other frameworks have the option to apply normalization across multiple axes, which leads to differences in `LayerNorm` on NCHW input by default. See Figure 3. Other frameworks can normalize over C, H and W, not just C as with MXNet Gluon.

Remember: `LayerNorm` is intended to be used with data in NTC format so the default normalization axis is set to -1 (corresponding to C for channel). Change this to `axis=1` if you need to apply `LayerNorm` to data in NHWC format.

Figure 3: `LayerNorm` on NCHW data | Figure 4: `LayerNorm` on NTC data 
- | - 
![alt](./imgs/NCHW_LN.png) | ![alt](./imgs/NTC_LN.png)
(e.g. batch of images) overriding the default with `axis=1` | (e.g. batch of sequences) using the default of `axis=-1`

As an example, we'll apply `LayerNorm` to a batch of 2 samples, each with 4 time steps and 2 channels (in NTC format).


```python
data = mx.nd.arange(start=0, stop=2*4*2).reshape(2, 4, 2)
print(data)
```

    
    [[[  0.   1.]
      [  2.   3.]
      [  4.   5.]
      [  6.   7.]]
    
     [[  8.   9.]
      [ 10.  11.]
      [ 12.  13.]
      [ 14.  15.]]]
    <NDArray 2x4x2 @cpu(0)>


With MXNet Gluon we can apply layer normalization with the `mx.gluon.nn.LayerNorm` block. We need to call `initialize` because `LayerNorm` has two learnable parameters by default: `beta` and `gamma` that are used for post normalization shifting and scaling of each channel.


```python
net = mx.gluon.nn.LayerNorm()
net.initialize()
output = net(data)
print(output)
```

    
    [[[-0.99997997  0.99997997]
      [-0.99997997  0.99997997]
      [-0.99997997  0.99997997]
      [-0.99997997  0.99997997]]
    
     [[-0.99997997  0.99997997]
      [-0.99997997  0.99997997]
      [-0.99997997  0.99997997]
      [-0.99997997  0.99997997]]]
    <NDArray 2x4x2 @cpu(0)>


We can see that normalization has been applied across all channels for each time step and each sample.

We can also check the parameters `beta` and `gamma` and see that they are per channel (i.e. 2 of each in this example).


```python
print('beta:', net.beta.data().asnumpy())
print('gamma:', net.gamma.data().asnumpy())
```

    beta: [ 0.  0.]
    gamma: [ 1.  1.]


##  Instance Normalization

Another less common normalization technique is called `InstanceNorm`, which can be useful for certain tasks such as image stylization. Unlike `BatchNorm` which normalizes across all samples of a batch per channel, `InstanceNorm` normalizes **across all spatial dimensions per channel per sample** (i.e. each sample of a batch is normalized independently).

Watch out: `InstanceNorm` is ideal is better suited to convolutional networks (CNNs) than recurrent networks (RNNs). We expect the input distribution to the recurrent cell to change over time, so normalization over time doesn't work well. LayerNorm is better suited for this case.

Figure 3: `InstanceNorm` on NCHW data | Figure 4: `InstanceNorm` on NTC data 
- | - 
![alt](./imgs/NCHW_IN.png) | ![alt](./imgs/NTC_IN.png)
(e.g. batch of images) using the default `axis=1` | (e.g. batch of sequences) overiding the default with `axis=2` (or `axis=-1` equivalently)

As an example, we'll apply `InstanceNorm` to a batch of 2 samples, each with 2 channels, and both height and width of 2 (in NCHW format).


```python
data = mx.nd.arange(start=0, stop=2*2*2*2).reshape(2, 2, 2, 2)
print(data)
```

    
    [[[[  0.   1.]
       [  2.   3.]]
    
      [[  4.   5.]
       [  6.   7.]]]
    
    
     [[[  8.   9.]
       [ 10.  11.]]
    
      [[ 12.  13.]
       [ 14.  15.]]]]
    <NDArray 2x2x2x2 @cpu(0)>


With MXNet Gluon we can apply instance normalization with the `mx.gluon.nn.InstanceNorm` block. We need to call `initialize` because InstanceNorm has two learnable parameters by default: `beta` and `gamma` that are used for post normalization shifting and scaling of each channel.


```python
net = mx.gluon.nn.InstanceNorm()
net.initialize()
output = net(data)
print(output)
```

    
    [[[[-1.34163547 -0.44721183]
       [ 0.44721183  1.34163547]]
    
      [[-1.34163547 -0.44721183]
       [ 0.44721183  1.34163547]]]
    
    
     [[[-1.34163547 -0.44721183]
       [ 0.44721183  1.34163547]]
    
      [[-1.34163547 -0.44721183]
       [ 0.44721183  1.34163547]]]]
    <NDArray 2x2x2x2 @cpu(0)>


We can also check the parameters `beta` and `gamma` and see that they are per channel (i.e. 2 of each in this example).


```python
print('beta:', net.beta.data().asnumpy())
print('gamma:', net.gamma.data().asnumpy())
```

    beta: [ 0.  0.]
    gamma: [ 1.  1.]

