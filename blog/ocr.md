
# Handwriting OCR: handwriting recognition and language modeling with MXNet Gluon

Author: Jonathan Chung, Applied scientist intern at Amazon.

Our pipeline to automatically recognize handwritten text includes: page segmentation [1] and line segmentation [2], followed by handwriting recognition is illustrated in Figure 1.

![Figure 1: Pipeline for handwriting recognition.](https://cdn-images-1.medium.com/max/12820/1*nJ-ePgwhOjOhFH3lJuSuFA.png)*Figure 1: Pipeline for handwriting recognition.*

In the previous blog post, we used an object detection paradigm to identify lines of text in the IAM dataset [2]. In the current blog post, I will discuss methods to recognize handwriting. This includes detecting characters given an image containing a line of text and denoising the output of the characters by using a language model.

## Methods

The pipeline of this component is presented in Figure 2. The input is an image containing a line of text and the output of the module is a string of the text.

![Figure 2: Pipeline of the handwriting detection and language model](https://cdn-images-1.medium.com/max/2294/1*RdDMALVLjD614nZuSewqHg.png)*Figure 2: Pipeline of the handwriting detection and language model*

### Handwriting detection

The handwriting detection takes the input image containing a line of text and returns a matrix containing probability of each character appearing. Specifically, the matrix is of size (sequence_length × vocab. of characters).

Previous works utilize multidimensional LSTMs to recognize handwriting [3] however more recent works suggest that similar performances can be achieved with a 1D-LSTM [4]. Briefly, [4] utilized a CNN for image feature extraction and fed the features into a bidirectional LSTM and trained the network to optimize the Connectionist Temporal Classification (CTC) loss (shown in Figure 3). Intuitively, the CNN generates image features that are spatially aligned to the input image. The image features are then sliced along the direction of the text and sequentially fed into an LSTM. This network is denoted as CNN-biLSTM. The main reason that the CNN-biLSTM was selected is because multidimensional LSTMs substantially more computationally expensive compared to the CNN-biLSTM.

![Figure 3: General framework of the CNN-biLSTM](https://cdn-images-1.medium.com/max/2000/1*KpP3hd55nZbc8qVQKt0XRg.png)*Figure 3: General framework of the CNN-biLSTM*

I extended the described implementation in [4] by providing multiple downsamples of the image features (Figure 4). Multiple downsamples were provided to assist in recognizing images of handwritten text that vary in size (e.g., lines that contain only 1 word vs lines that contain 7 words). Note that a pre-trained res-net was used as a image feature extractor.

![Figure 4: Our implementation of the downsampled CNN-biLSTM](https://cdn-images-1.medium.com/max/2000/1*JTbCUnKgAySN--zJqzqy0Q.png)*Figure 4: Our implementation of the downsampled CNN-biLSTM*

The output of the CNN-biLSTM is fed into a decoder to predict probability distribution over the characters for each vertical slice of the image. In the next section, methods to extract the most probably sentence from the matrix using a language model are discussed.

## Language models

We explored three methods to extract a string of words given a matrix of probabilities are explored: 1) greedy search, 2) lexicon searching, and 3) beam searching + lexicon searching with a simple language model.

### Greedy search

The greedy search method simply iterates over each time-step and obtains the most probable character at each time step:

<iframe src="https://medium.com/media/8b022e14849e1688ec995028697017d4" frameborder=0></iframe>

This method doesn’t use any external language models or dictionary to alter the results.

### Lexicon searching

The lexicon search model use the output of the greedy method and attempts to match each word with a dictionary. Specifically, the following procedures were taken: 1) Decontraction of the string, 2) Tokenization of the string, 3) for each word: 3.1) Check if the proposed word is an English word, if not, suggest possible words, and 3.2) Choose the word with the shortest weighted edit distance, 4) Contract texts back to its original form. Each component are described and the pseudocode is provided below.

<iframe src="https://medium.com/media/cad9d335e741fb7d61fe23e90a225875" frameborder=0></iframe>

1. **Decontraction of the string**

Contractions are a shorted version of a word or a spoken form of a word. For example, did not → didn’t, might have → might’ve, for a more comprehensive list see [here](https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions)). In order to account for contractions, [pycontractions](https://github.com/ian-beaver/pycontractions) was used. This library provides contractions and decontractions of the string. For decontracting ambiguous cases (e.g., ain’t → am not/are not), the library utilizes a language model (google news) to assess the correct case.

**2. Tokenization of the string**

Tokenization separates a string containing a sentence into distinct words. The [tokenize](https://www.nltk.org/api/nltk.tokenize.html) package from the nltk toolkit was used to perform this. This was followed by iterating through each word.

**3.1 Suggest words if it’s not a word**

For each word that was tokenized, the [pyenchant](https://github.com/rfk/pyenchant) module is used to check if the word is an actual English word. If the word is an actual word, then no changes will be made, otherwise, the [norvig spell checker](http://norvig.com/spell-correct.html) was used to provide suggestions for possible words.

**3.2 Select the suggested word with the shortest weighted edit distance**

Given a list of suggested words, the [weighted edited distance](https://github.com/infoscout/weighted-levenshtein) between the given word and the suggested words were calculated. Specifically, the edit distance was weighted on the differences between the handwritten characters.

![Figure 5: Examples of characters that are similar in terms of handwriting](https://cdn-images-1.medium.com/max/2000/1*v-NNY8Z8eZkRivH4O_HgPA.png)*Figure 5: Examples of characters that are similar in terms of handwriting*

Characters that are shown in Figure 5 should have lower weights compared to characters that are visual different (e.g., “*w*” and “*o*”) and the visually different characters should have large weights.

The character similarity was modeled using the output of the CNN-biLSTM given the training examples. The frequency of differences (insertions, deletions, and substitutions) between actual words and the predicted words was counted. The larger the frequency of a mistake, the more visually similar characters are. Therefore, the smaller the weight should be. The main issue is that the weighted edit distance is limited to substitutions of one character to another. In handwriting recognition, it is common for compounds to characters to look alike (e.g., “*rn*” and “*m*”).

**4. Contract texts back to its original form**

This step simply detokenizes and contracts the words (if it was originally decontracted).

As a whole, the lexicon spell checker works best for predicted strings that are very close to the actual value. However, if the output of the greedy algorithm doesn’t provide close enough string proposals, the lexicon spell checker can actually reduce accuracies instead of improving it. A final check to ensure that the input and output sentences are similar was performed.

### Beam searching + lexicon search + language model

In order to alleviate the issues of getting poor proposals from the greedy algorithm, one can iterate through the probability matrix to obtain multiple proposals. However, ranking all possible proposals is computationally expensive. Graves et al. [5] proposed that the beam search algorithm can be used to generate generate *K* proposals of strings given the probability matrix.

In our implementation, the *K* sentences are then fed into the lexicon spell checker to ensure that words can be found in a dictionary (otherwise it will appear as <unk>). Each *K* proposal is then tokenized and sequentially fed into a language model to evaluate the perplexity of each sentence proposal. The sentence proposal with the lowest perplexity was chosen.

## Results

The algorithm was qualitatively and quantitatively evaluated. Examples of the predicted text from four forms are shown in Figure 6.

![Figure 6: Qualitative results of the handwriting recognition and language model outputs (method *a*, *b*, and *c*)](https://cdn-images-1.medium.com/max/3046/1*8lnqqlqomgdGshJB12dW1Q.png)*Figure 6: Qualitative results of the handwriting recognition and language model outputs (method *a*, *b*, and *c*)*

The greedy, lexicon search, and beam search outputs present similar and reasonable predictions for the selected examples. In Figure 6, interesting examples are presented. The first line of Figure 6 show cases where the lexicon search algorithm provided fixes that corrected the words. In the top example, *“tovely”* (as it was written) was corrected *“lovely”* and *“woved”* was corrected to *“waved”. *In addition, the beam search output corrected “*a*” into “*all*”, however it missed a space between “*lovely*” and “*things*”. In the second example, “*selt*” was converted to “*salt*” with the lexicon search output. However, “*selt*” was erroneously converted to “*self*” in the beam search output. Therefore, in this example, beam search performed worse. In the third example, none of the three methods significantly provided comprehensible results. Finally, in the forth example, the lexicon search algorithm incorrectly converted “*forhim*” into “*forum*”, however the beam search algorithm correctly identified “*for him*”.

Quantitatively, the greedy algorithm had a mean character error rate (CER) = 18.936 whereas the lexicon search had CER = 18.897. Without the weighted edit distance, the lexicon search had CER = 19.204, substantially reducing the performance of algorithm. As presented in Figure 6, the CER improvement between using the greedy algorithm and the lexicon search was minimal. As expected, the beam search algorithm out performed the greedy and the lexicon search results with CER = 18.840.

Check out [this](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/handwriting_ocr.ipynb) notebook for a reference implementation

## References

[1] Page segmentation with Gluon, [https://medium.com/apache-mxnet/page-segmentation-with-gluon-dcb4e5955e2](https://medium.com/apache-mxnet/page-segmentation-with-gluon-dcb4e5955e2)

[2] Handwriting OCR: Line segmentation with Gluon, [https://medium.com/apache-mxnet/handwriting-ocr-line-segmentation-with-gluon-7af419f3a3d8](https://medium.com/apache-mxnet/handwriting-ocr-line-segmentation-with-gluon-7af419f3a3d8)

[3] Graves, A., & Schmidhuber, J. (2009). Offline handwriting recognition with multidimensional recurrent neural networks. In *Advances in neural information processing systems* (pp. 545–552).

[4] Puigcerver, J. (2017, November). Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?. In *Document Analysis and Recognition (ICDAR), 2017 14th IAPR International Conference on* (Vol. 1, pp. 67–72). IEEE.

[5] Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006, June). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In *Proceedings of the 23rd international conference on Machine learning* (pp. 369–376). ACM.
