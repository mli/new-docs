
# GluonNLP — Deep Learning Toolkit for Natural Language Processing

Original author: Sheng Zha
Translated from https://zh.mxnet.io/blog/gluon-nlp
Editors: Hao Jin, Thomas Delteil

![](https://cdn-images-1.medium.com/max/3252/0*BHqhpCQMCEV3YD0b.png)

*Why are the results of the latest models so difficult to reproduce?* *Why is the code that worked fine last year not compatible with the latest release of my deep learning framework?* *Why is a baseline benchmark meant to be straightforward so difficult to set up?* *In today’s world, these are the challenges faced by Natural Language Processing (NLP) researchers.*

Let’s take the case of a hypothetical PhD student. Let’s call him Alexander, who just began his research on Neural Machine Translation (NMT). One morning he came across one of the most popular papers from Google: “Attention is all you need”, which introduces the Transformer model, solely based on the attention mechanism. Impressed by the results, Alex performed a quick search on Google that informed him that *Tensorflow’*s *Tensor2Tensor* package already contained an implementation of this model. Cries of joy resonated in his lab’s cubicle; he should be able to reproduce the results within the afternoon and enjoy a well-deserved relaxing evening watching the world cup with his friends at the pub! Or so he thought…

Soon, our friend found out that the official implementation’s hyperparameters differed widely from the ones mentioned in the paper. His supervisor suggested, “Just tweak the parameters a bit, run and see, it should be fine” — shouldn’t it?

![](https://cdn-images-1.medium.com/max/2000/0*LOo83HPsoANphHe6.jpeg)

Fast-forward three days… All the GPUs in the lab cluster are starting to smoke after being run at 100% capacity continuously. His labmates forcefully logged him out of his Linux shell and told him to go home and have some rest. Our unlucky hero did so, and resolved to go to *Tensor2Tensor’*s Github to report the issue, to see if anybody had gone through the same ordeal as him. Soon, several other users replied that they had indeed the same problem, but they had found no solution for it.

![](https://cdn-images-1.medium.com/max/2956/0*I9swL5vZ6rs4oLoL.png)

Half a month passed… Finally, the project maintainer appeared and replied that he would look into it.

![](https://cdn-images-1.medium.com/max/3132/0*nyIFkBvFAdokyV0F.png)

Three months passed… Alex is still asking helplessly: “Is there any progress?”

![](https://cdn-images-1.medium.com/max/3060/0*SH7G_ejxOdbJWRfd.png)

After working on deep learning and Natural Language Processing (NLP) at Amazon Web Services (AWS) for a while, I found that this is not an isolated example. Reproducing NLP models can be a lot harder than computer vision ones. The data pre-processing pipeline involves significantly more steps, and the models themselves have a lot more moving parts. For example, one needs to be careful with the following items, to name a few:

* String encoding/decoding and Unicode format

* Parsing and tokenization

* Text data that can be from various languages following different grammatical and syntactic rules

* Left-to-right or right-to-left reading order​

* Word embedding

* Input padding

* Gradient clipping

* Dealing with variable-length input data and states.

From the loading of the training dataset to the output of the BLEU score on the testing set, there will be a thousand opportunities to do something wrong. If the proper tools are not in place, every time you start a new project, you risk getting acquainted with a whole new set of pitfalls and previously unknown issues.

Several MXNet contributors and I once brought up all these different issues we had encountered during our careers in NLP, and everyone had similar war stories and anecdotes to share that strongly resonated with one another. We all agreed that although NLP is hard, we still wanted to do something about it! We decided to develop a *toolkit* that can help you easily reproduce the latest research results and easily develop new models in Gluon. This team was composed of Xingjian Shi (@[sxjscience](https://github.com/sxjscience)), Chenguang Wang (@[cgraywang](https://github.com/cgraywang)), Leonard Lausen (@[leezu](https://github.com/leezu)), Aston Zhang (@[astonzhang](https://github.com/astonzhang)), Shuai Zheng (@[szhengac](https://github.com/szhengac)), and myself (@[szha](https://github.com/szha)). [GluonNLP](https://gluon-nlp.mxnet.io/) was born!
> **✘ Symptom:** Natural language processing papers are difficult to reproduce. The quality of open source implementations available on Github varies a lot, and maintainers can stop supporting the projects

**√ GluonNLP prescription: **Reproduction of latest research results. Frequent updates of the reproduction code, which comes with training scripts, hyper-parameters, runtime logs etc.

Being unable to reproduce a result that you were once able to get is a terrible, though unfortunately not uncommon experience. In theory, your code will not deteriorate by itself over time, whilst in practice we all have had such experience. One of the main reasons is that the APIs provided by the underlying deep learning library will change over time.

In order to study that phenomenon, we ran the following query in Google search:— **“X API break”** .

    ┌────────────┬────────────────┐
    │     X      │ Search Results │
    ├────────────┼────────────────┤
    │ MXNet      │ ~17k           │
    │ Tensorflow │ ~81k           │
    │ Keras      │ ~844k          │
    │ Pytorch    │ ~23k           │
    │ Caffe      │ ~110k          │
    └────────────┴────────────────┘

Although the MXNet community is putting a lot of effort into keeping the API breaking changes to a minimum, this search nonetheless returned 17k results. To prevent our package from running into such problems, we are especially attentive to that during our development. Every training script is integrated into Continuous Integration (CI)to catch any regression and avoid creating bad surprises for our users.
> **✘ Symptom:** Reproduction code is sensitive to API changes thus has limited shelf-life.

**√ GluonNLP Prescription: **Automated testing of the training and evaluation scripts, to adapt to the latest APIs changes as early as possible.

While working on some NLP services last year, I ended up with five implementations of the Beam Search algorithm, each slightly different from one another. The reason behind this was that the interface of the module was rushed due to project deadlines and thus poorly designed in the first place. When facing different use cases, the easiest thing to do is to copy and tweak each of them. As a result, I ended up with my five versions of Beam Search differing only in the score and step function.

Creating a reusable and easily extensible interface usually requires a lot of effort on studying various usages and heated discussions among several developers. In GluonNLP, we focus not only on reproducing existing examples but also developing easy-to-use, extensible interfaces and tools based on these examples, to pave the way for future research.
> **✘ Symptom: **Copy-pasting code from older projects to new ones with minor modifications, as short-term solutions, due to tight deadlines without careful design of interfaces resulting in hard-to-maintain code.

**√ GluonNLP prescription: **Easy-to-use and extensible interfaces based on GluonNLP’s squad’s research on various usages from different projects.

Recently, I have been working on a new project and found that a recurring issue is that a lot of useful resources are not centralized. Everyone knows that pre-trained word embeddings and language models are useful for a wide range of applications. However picking which pre-trained model to use for a given task requires a lot of manual experimentation. Developers often need to install many different tools when doing this exploratory work. For example, Google’s *Word2vec* has a dependency on *gensim*, whilst Salesforce’s *AWD* language model is implemented with *PyTorch*, and does not provide a pre-trained model. Facebook’s *Fasttext* is a self-developed independent package. In order to be able to use these resources in their own preferred framework, users often need to spend significant effort on setting up the environment and converting data from one format to another.

We all know that resource sharing is one of the most important features of a community. In [GluonNLP](http://gluon-nlp.mxnet.io/), we not only hope to provide tools and a community for NLP enthusiasts, but also providing them with easy access to resources by integrating the ones already available on various other platforms, thus making GluonNLP a one-stop shop.
> **✘ Symptoms: **NLP resources are scattered. To complete one project you may have to depend on multiple packages.

**√ GluonNLP prescription: **Aggregation and redistribution of useful public resources. One-click download of pre-trained word embeddings, pre-trained language models, common benchmark datasets and pre-trained models for various applications.

## Enough talking, show me the code!

<iframe src="https://medium.com/media/3a261f3eaba249fa8b03f0b557e12e6b" frameborder=0></iframe>

In this example we’re using GluonNLP to load the GloVe word embeddings and word embeddings from the pre-trained AWD language model, then comparing their performance on measuring similarities. We do that by comparing the cosine similarity between the embeddings of the words ‘baby’ and ‘infant’. From the results it seems like GloVe embedding is better at capturing semantic similarity.

## Where is the project?

The latest GluonNLP is available at [gluon-nlp.mxnet.io](http://gluon-nlp.mxnet.io) and can be installed with pip install gluonnlp and the latest version of [MXNet](http://mxnet.io/).

As of v.0.3.2, GluonNLP features:

* Pre-trained models: over 300 word-embedding (GloVe, FastText, Word2vec), 5 language models (AWD, Cache, LSTM).

* Neural Machine Translation (Google NMT, Transformer) model training.

* Word Embedding training for Word2vec and FastText, including unknown word embedding interpolation using subword embeddings.

* Flexible data pipeline tools and many public datasets.

* NLP examples such as sentiment analysis.

We will continue to add [new features and models](https://github.com/dmlc/gluon-nlp/releases/latest) in the following releases. If you are interested in specific models or have some feedback, find us on [Github](https://github.com/dmlc/gluon-nlp)!
