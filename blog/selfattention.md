
# Sentiment Analysis via Self-Attention with MXNet Gluon

Recently, the “attention” mechanism has become one of the core techniques in Deep Learning throughout a variety of domains after its huge initial success in Neural Machine Translation (NMT) [Cho et al., 2015 and references therein]. It improved over existing Recurrent Neural Network (RNN) based NMT algorithms significantly by relaxing the assumption that all the information from the input sentence should be compressed into a single hidden state vector.

![[[Bahdandau et al., 2015]](http://arxiv.org/abs/1409.0473)](https://cdn-images-1.medium.com/max/2000/0*rbcu-PtTOYAqMy5a.png)*[[Bahdandau et al., 2015]](http://arxiv.org/abs/1409.0473)*

In the graph above, RNNsearch-50 is the result of the NMT model equipped with the soft attention mechanism and we can see that BLEU score does not drop as the input sentence gets longer. The authors believe that the attention mechanism helped to convey long-term information that cannot be retained with smaller hidden vector representations.
> NOTE: BLEU score is a measure to evaluate the quality of translated text. For more information, please refer to [WIKI](https://en.wikipedia.org/wiki/BLEU) page.

The attention mechanism can also be thought of as a great way to reconcile CNN based methods focusing on a finite-sized receptive field for a given sentence and those based on RNN that can theoretically retrieve information from the entire input sentence.

**Self-Attention (SA)**, a variant of the attention mechanism, was proposed by [Zhouhan Lin, et. al (2017)](https://arxiv.org/abs/1703.03130) to overcome the drawback of RNNs by allowing the attention mechanism to focus on segments of the sentence, where the relevance of the segment is determined by the contribution to the task. Self-attention is a relatively simple-to-explain mechanism. This is a welcome change to understand how a given deep learning model works, as a lot of previous NLP architectures are known for their black-box and hard-to-interpret natures.

![Zhouhan Lin, et. al (2017)](https://cdn-images-1.medium.com/max/2940/1*6c4-E0BRRLo197D_-vyXdg.png)*Zhouhan Lin, et. al (2017)*

Here is a brief summary of what the authors proposed in the paper:

From part (a) in the above diagram, we can see the entire architecture of the self-attention model. The embedded tokens (*w*’s in the above) are fed into bidirectional LSTM layers (*h*’s). Hidden states are weighted by an attention vector (*A*’s) to obtain a refined sentence representation (*M* in the above) that is used as an input for the classification.

How did we get the attention weights? It is illustrated in part (b) of the diagram, proceeding from top to bottom. To begin with the collection of hidden states, it is multiplied by a weight matrix, and is followed by tanh layer for non-linear transformation. And then another linear transformation is applied to the output with another weight matrix to get the pre-attention matrix. A softmax layer, which is applied to the pre-attention matrix in the row-wise direction, making its weights looking like a probability distribution over the hidden states.

The original paper applied the SA mechanism on top of two bidirectional LSTM layers, but various configurations can be used. We will now dive into more details about this architecture and discuss how we can implement it with Gluon to perform sentiment analysis of movie reviews.

Let’s first briefly review different sentence representation methods:

## Sentence representation

Human-readable sentences need to be translated into machine-readable ones for NLP tasks, including sentiment analysis. This can be conceptually divided into two stages: one is to single out tokens appearing in a given sentence (tokens can be either ***word*** or ***character**, or even **bytes***) and the other is to represent the entire sentence as a vector or matrix.

One-hot encoding is one of the easiest way to quantify tokens, but it frequently results in a huge vector depending on the size of corpus, consisting of bunch of 0’s and a 1 to specify the corresponding index in a given vocabulary. Therefore, one-hot vector representations are very inefficient in terms of memory. If we work with words as tokens, it gets even worse since the vocabulary grows as the dataset gets bigger, or need to be capped and information is lost.

![One hot vector example](https://cdn-images-1.medium.com/max/3468/1*kIN812aiq7jqpCmlp-iQkQ.png)*One hot vector example*

To prevent this, we represent tokens as embedding vectors, which reside in much smaller dimensional space represented with real numbers rather than 0s and 1s. Most NLP networks contain embedding layers at the very beginning of the network.

![Token ‘Love’ from a vocabulary of size 5 embedded in a 3-dimensional space](https://cdn-images-1.medium.com/max/2000/1*cBQNgPOEmmTrGfw3dXhLzw.png)*Token ‘Love’ from a vocabulary of size 5 embedded in a 3-dimensional space*

Pre-trained embedding layers can be set directly in the network rather than going through the process of learning brand new embedding representations, but whether or not this is beneficial needs to be evaluated on a case-by-case basis.

Once tokens are quantized, we are ready to represent sentences with those tokens. As a naive first step, we can think of summing all those individual token representations up (we can easily do this, because they have the same shape) losing the token’s ordering information. This performs well on text classification tasks, however it doesn’t learn the semantic information in sentences and simply relies on token statistics.

![Sentiment analysis based on average pooling](https://cdn-images-1.medium.com/max/3000/1*BrPSQj4AaOFvStLkcGRL0A.png)*Sentiment analysis based on average pooling*

To mimic the way human read sentences and capture the sequence information, there are several deep learning architecture available such as RNN, CNN and their combination. RNNs accumulates sequential token information presented in the sentence in their hidden states.

![Sentiment analysis based on RNN using hidden state information at the last time step](https://cdn-images-1.medium.com/max/2000/1*7FI4Fri3DWXjsoowB_mwMw.png)*Sentiment analysis based on RNN using hidden state information at the last time step*

Specifically, the above figure depicts an architecture where only the last hidden state (at the end of each sentence) is used for classification. There are other techniques that make use all the intermediary hidden states information, through summation or averaging. In the case of sentiment analysis, bidirectional LSTMs are frequently used to capture the right-to-left and left-to-right relationships between tokens in the sentence and to limit the weight given to the last token as the first tokens are “forgotten” by the network. We employ a bidirectional LSTM in this article.

## Self-Attention

The Self-Attention mechanism is a way to put emphasis on tokens that should have more impact on the final result. [Zhouhan Lin, et. al (2017)](https://arxiv.org/abs/1703.03130) proposed the following architecture for self-attention as is shown in the following figure. With *u* the dimension of the hidden state of a LSTM layer, we have *2*u* as dimension of hidden state since we use a **bidirectional LSTM.** As we have *n* tokens in a sentence, there are *n* hidden states of size *2*u*. A linear transformation from *2u*-dimensional space to *d*-dimensional one is applied to the *n* hidden state vectors. After applying *tanh* activation, another linear transformation from *d*-dimension to *r*-dimension is applied to come up with *r* dimensional attention vector per token. Now, we have r attention weight vectors of size n (denoted as *A* in red box from the figure below), and we use them as weights when averaging hidden states, to end up with r different weighted averages of 2*u vectors (denoted as M in the figure from the original paper).

![Self attention architecture and its implementation using dense layer](https://cdn-images-1.medium.com/max/2000/1*dtC80EsitkHgK421wqJijw.png)*Self attention architecture and its implementation using dense layer*

When it comes to implementation, we can deal with matrix multiplication as a part of graph. If we consider each time step as an independent observation, we can consider each linear transformation as a fully connected layer without bias. In that case, batch size would be inflated n times. We have to use reshaping techniques for this as shown in the following code snippet:

<iframe src="https://medium.com/media/d590e66a061566d2741d5446cbb15902" frameborder=0></iframe>

## Regularization

The authors also introduced a penalty term based on the self-attention matrix as follows:

![Penalty term for regularizing similarity of r-hops of attentions](https://cdn-images-1.medium.com/max/2000/1*XPDCOnQy4w69tjaYL2TZog.png)*Penalty term for regularizing similarity of r-hops of attentions*

This prevents multiple attention vectors from being similar or redundant. This penalty encourages the self-attention matrix to have large values on its diagonal and it lets single attention weights for a given token dominates other (r-1) attention weights.

We used spacy for data-processing and seaborn for visualization. The entire code can be found in [[Sentiment Analysis by Self-Attention](https://github.com/kionkim/stat_analysis/blob/master/notebooks/text_classification_RNN_SA_umich.ipynb)].

## Results

In this experiment, we limit the length of each sentence to 20 tokens. As hyperparameters, we used *d=*10 and *r=*5. Therefore, once trained, we end up with 5 attention weight vectors capturing different aspect of the sentence. For illustration purposes, we averaged the 5 weights and applied a softmax filter again to get a probability distribution over the tokens (sum to 1).

We used a simple classifier with two fully connected layers and a binary classification entropy loss. Other miscellaneous parameters are given in the example code. Here are the visualizations for 10 positive and negative reviews with attention weights colored as background. Greens get more attention than reds.

![10 positive reviews with attention weights](https://cdn-images-1.medium.com/max/3000/1*JQx2ICjmduz5k7L96KWMHw.png)*10 positive reviews with attention weights*

For the positive reviews, the algorithm paid attention to positive words such as ‘awesome’, ‘love’, and ‘like’.

![10 negative reviews with attention weights](https://cdn-images-1.medium.com/max/3616/1*D4Q_zhfq_GHyxDMhxFKLlA.png)*10 negative reviews with attention weights*

For negative reviews, the algorithm focused on negative words such as ‘suck’, ‘hate’ , ‘stupid’ and so on.

In our experiment, there are 28 out of 3,216 sentences misclassified. Let’s have a look at one of them:

    like mission impossible but hate tom cruise get that straight update day in a row like magic and shit

We can see that the sentence includes both positive and negative words such as ‘like’, ‘hate’, ‘shit’, ‘magic’. Understandably the model got confused by this review that mixes language elements that are positive and others that are negative.

## Conclusion

As attention mechanisms are becoming more and more prevalent in Deep Learning research, it is crucial to understand how they work and how to implement them. We hope that this article helped you be more familiar with the self-attention mechanism!
