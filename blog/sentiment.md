
# Let Sentiment Classification Model speak for itself using Grad CAM

Deep learning models are known for being black box models. However, according to our experience, recent developments in explainable methods reduced such a biased view of Deep Learning. In this article, we show how to visualize neural network’s decision-making process based on the movie review sentiment classification model.

In this example, we use IMDB reviews data. We take training and test sets as is, and we convert ratings to sentiment labels (1: positive(if rating > 5), 0: negative (if rating ≤ 5)).

### Data Preprocessing

The preprocessing of IMDB data extracts the sentence unit tokens and removes the stop words and inputs it through embedding. Therefore, the preprocessing section for input data is the same as the one introduced [here](https://gluon-nlp.mxnet.io/api/notes/data_api.html)¹, except for the following lines.

    import spacy

    nlp = spacy.load("en")

    def preprocess(x):
        data, label = x
        label = int(label > 5)
        data = nlp(data)
        data = length_clip(**[token.lemma_ for token in data if not token.is_stop]**)
        return data, label, x

We used pre-trained embedding as below. The use of pre-trained embedding has a great influence on convergence speed.

    fasttext_en = gnlp.embedding.create('fasttext', source='wiki.en')

    vocab.set_embedding(fasttext_en)

### Model

The model architecture was based on the neural network with three convolutions as follows. The layer to which Grad CAM is applied is ‘self.conv1’.

    class SentClassificationModel(gluon.Block):

        def __init__(self, in_vocab_size, out_vocab_size, **kwargs):
            super(SentClassificationModel, self).__init__(**kwargs)
            with self.name_scope():
                self.embed = nn.Embedding(input_dim=in_vocab_size,
                                          output_dim=out_vocab_size)
                self.conv1 = nn.Conv1D(32, 1, padding=0)
                self.conv2 = nn.Conv1D(16, 2, padding=1)
                self.conv3 = nn.Conv1D(8,  3, padding=1)
                self.pool1 = nn.GlobalAvgPool1D()
                self.pool2 = nn.GlobalAvgPool1D()
                self.pool3 = nn.GlobalAvgPool1D()
                self.dense = nn.Dense(2)
                self.conv1_act = None

        def forward(self, inputs):
            em_out = self.embed(inputs)
            em_swaped = nd.swapaxes(em_out, 1,2)
            self.conv1_act = self.conv1(em_swaped)            
            conv1_out = self.pool1(self.conv1_act)
            conv2_ = self.conv2(em_swaped)
            conv2_out = self.pool2(conv2_)
            conv3_ = self.conv3(em_swaped)
            conv3_out = self.pool3(conv3_)
            cated_layer = nd.concat(conv1_out, conv2_out, 
                                    conv3_out , dim=1)
            outs = self.dense(cated_layer)
            return outs

### Training

    epochs = 2

    tot_train_loss = []
    for e in range(epochs):
        train_loss = []
        for i, (data, label) in enumerate(tqdm(train_dataloader)):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data)
                loss_ = loss(output, label)
                loss_.backward()
            trainer.step(data.shape[0])
            curr_loss = nd.mean(loss_).asscalar()
            train_loss.append(curr_loss)
        train_acc = evaluate(model, train_dataloader, ctx)
        test_acc  = evaluate(model, test_dataloader, ctx)
        print("Epoch %s. Train Loss: %s, Train Accuracy : %s, 
              Test Accuracy : %s" % 
              (e, np.mean(train_loss), train_acc, test_acc))    
        tot_train_loss.append(np.mean(train_loss))

    100%|██████████| 250/250 [00:13<00:00, 18.97it/s]
    Epoch 0. Train Loss: 0.48509014, Train Accuracy : 0.89392, Test Accuracy : 0.8668
    100%|██████████| 250/250 [00:12<00:00, 19.78it/s]
    Epoch 1. Train Loss: 0.26659298, Train Accuracy : 0.92976, Test Accuracy : 0.87604

![AUC : 0.9455451392](https://cdn-images-1.medium.com/max/4800/1*2geDWxF_zxbOFT4ztmpYcg.png)*AUC : 0.9455451392*

### Grad CAM

After training, it’s time to check how the model decides the sentiment classification by applying Grad CAM to the model.

![[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)](https://cdn-images-1.medium.com/max/2832/1*KSTv_NEaeQ1cj_mrEnFXrw.png)*[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)*

Although the method proposed in [Grad CAM](https://arxiv.org/abs/1610.02391) paper was used on images, Convolution-based techniques are also used for the problems of the text classification modeling area. The Grad CAM scheme is a method of assigning a score to each entry using the backpropagation-based filter gradient and convolution activation values that cause the model to determine that the review is positive or negative. This type of convolution visualization actually uses the gradient ascent technique, but it is a bit easier to understand with the idea of what to maximize. Indeed, in a deep learning process, weighted delta updates of “gradient ×-1 ×learning_rate” are done to reduce the loss. Conversely, we add more delta to the area to be visualized and what we want to see gets highlighted

In summary, it is the core of the Grad CAM to try to interpret the model by reversing the learning process. Here we try to analyze the individual layer of the 1D convolution selected in the model and weight the results from the layer activations to see how individual word entries are used to classify a review. This allows us to verify that the model is created with a common sense and even help us to debug the model.

![[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)](https://cdn-images-1.medium.com/max/2000/1*E2tEjAI5xj1xxXnlylfQaA.png)*[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)*

The formula above shows how global average pooling is performed on the gradient of the k-th feature map . This implies the importance of k-th feature map for target class c.

![[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)](https://cdn-images-1.medium.com/max/2000/1*lJsbFBhY2dqnVUSL3ZQmaQ.png)*[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)*

After that we take weighted average of the activation for each feature map by multiplying importance of the k-th feature map.

As we can access gradients and feature maps with Gluon very easily, the following lines of code will be enough to implement the algorithm.

    def grad_cam_conv1D(model, x, y, loss, ctx):
        with autograd.record(train_mode=False):
            output = model(nd.array([x,],ctx=ctx))
            loss_ = loss(output, nd.array([y,],ctx=ctx))
            output = nd.SoftmaxActivation(output)
            loss_.backward()
        acts = model.conv1_act
        #a_k^c
        global_avg_grad = nd.mean(model.conv1.weight.grad(), axis=(1,2))
        #L_{Grad-CAM}^c
        for i in range(acts.shape[1]):
            acts[:,i,:] *= global_avg_grad[i]
        heat = nd.relu(nd.sum(acts, axis=1))
        return(heat.asnumpy()[0], loss_)

### Test

Let’s see which tokens have a significant impact on class for the following randomly selected negative review.

    ["Pretty bad PRC cheapie which I rarely bother to watch over again, and it\'s no wonder -- it\'s slow and creaky and dull as a butter knife. Mad doctor George Zucco is at it again, turning a dimwitted farmhand in overalls (Glenn Strange) into a wolf-man. Unfortunately, the makeup is ...."]

![](https://cdn-images-1.medium.com/max/6000/1*7IDzYCBBjP2Pb9eVtmIcmg.png)

We can see that words such as ‘**dull, bad, unwatchable**, …’ have influenced the model to judge it as a negative review.

Implementation details can be found [here](https://github.com/haven-jeon/grad_cam_gluon/blob/master/text_grad_cam_imdb.ipynb)².

### Conclusion

* Grad CAM can be used not only for images but also for the interpretation of text classification models.

* Gluon provides a very intuitive interface for extracting activations and gradients.

[1] [https://gluon-nlp.mxnet.io/api/notes/data_api.html](https://gluon-nlp.mxnet.io/api/notes/data_api.html)
[2] [https://github.com/haven-jeon/grad_cam_gluon/blob/master/text_grad_cam_imdb.ipynb](https://github.com/haven-jeon/grad_cam_gluon/blob/master/text_grad_cam_imdb.ipynb)
