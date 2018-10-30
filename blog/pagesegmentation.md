
# Page segmentation with Gluon

Author: Jonathan Chung, Applied scientist intern at Amazon.

Despite living in the 21st century, a large portion of everyday documents are still handwritten. Many school notes, doctor notes, and historical documents are handwritten.

![Figure 1: Examples of handwritten texts](https://cdn-images-1.medium.com/max/3554/1*T4pXSTfsz9-vWgqVquLwfg.png)*Figure 1: Examples of handwritten texts*

Archiving the handwritten documents is essential but it is usually limited to storing high resolution images of the documents. As the textual information in the documents is difficult to recognise, people resort to storing manually transcribed text of the handwriting [1].

There are currently considerable efforts to develop automated methods to transcribe handwritten text. Usually, the data flow consists of performing page segmentation to identify regions of texts within a document then running handwritten text recognition [1].

The main focus of this blog post is to describe page segmentation methods. In future posts, handwritten text recognition using recurrent neural networks (RNN) will be described (similar to [2, 3]).

## Page segmentation

![Figure 2: Example of a document from the IAM dataset with a bounding box around the handwritten text](https://cdn-images-1.medium.com/max/2000/1*ykwPJSZZW-VSgrTVxN84BA.png)*Figure 2: Example of a document from the IAM dataset with a bounding box around the handwritten text*

Page segmentation has been widely studied in historical documents where documents are segmented into decoration, background, text block, and periphery. Traditionally, handcrafted features are used for segmentation but recently, Chen et al. [1] demonstrated that convolutional neural networks (CNNs) in an encoder-decoder architecture can automatically learn high-level feature representations of historic documents. The feature representations are fed into an SVM classifier to learn the segmentations of the document.

Here we focus on the segmentation of handwritten texts from the IAM dataset [4]. Documents from the IAM dataset contain a printed portion and handwritten portion. The goal of this algorithm is to fit a bounding box around the handwritten portion of the document (example shown in Figure 2).

## Methods

Two methods of obtaining the bounding box were explored: handcrafted features using the Maximally Stable Extremal Regions (MSERs) algorithm and using a deep CNN approach.

### MSERs algorithm approach

The MSERs algorithm was used to detect “blobs” on the image which correspond to text on the images. The detected regions are post-processed to identify continuous regions of text with the following algorithm:

For *i* in iteration:

1. Expand the bounding boxes in all directions by a fraction denoted by *Δ*

1. Merge all the bounding boxes that overlap a percentage denoted by *I*

Here are some of the results.

![Figure 3–1: Region proposal reduction algorithm (iterations=4, *Δ*=1.2, *I*=0.1)](https://cdn-images-1.medium.com/max/5468/1*H8jLNVgp5ka_gSNGdelH-w.png)*Figure 3–1: Region proposal reduction algorithm (iterations=4, *Δ*=1.2, *I*=0.1)*

The MSERs algorithm was previously successful for detecting blocks of printed text [5–6]. However, when the parameters (*Δ* and *I*) were not carefully tuned for the specific document, the algorithm fails to detect the passage. When the same image (in Figure 3–1) was used with different parameters, the algorithm fails to detect continuous regions of text (shown in Figure 3–2).

![Figure 3–2: Region proposal reduction algorithm (iterations=5, *Δ*=1.1, *I*=0.1)](https://cdn-images-1.medium.com/max/6808/1*-v-siX99R4IL7DnVIQdM3A.png)*Figure 3–2: Region proposal reduction algorithm (iterations=5, *Δ*=1.1, *I*=0.1)*

Handwritten text is more diverse compared to printed text where different people can have different writing styles, distance between letters, etc. After our experimentation, we found that it’s difficult to obtain parameters that can generalise between different individuals. We therefore decided to implement a CNN approach to paragraph segmentation.

Check out [this Jupyter Notebook](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/master/paragraph_segmentation_msers.ipynb) for a reference implementation of the MSER algorithm used above.

## Deep CNN approach

![Figure 4: Deep convolutional neural network used to obtain the segment the handwritten passages](https://cdn-images-1.medium.com/max/15336/1*AggJmOXhjSySPf_4rPk4FA.png)*Figure 4: Deep convolutional neural network used to obtain the segment the handwritten passages*

The deep CNN (shown in Figure 4) was written using [Apache MXNet](https://mxnet.incubator.apache.org/) and takes the IAM document as an input and predicts the bounding box of the handwritten passage. The network was initially trained to minimise the mean squared error of the predicted and actual bounding boxes. In Figure 5, the bounding boxes of eight images are shown as training progresses using [MXBoard](https://github.com/awslabs/mxboard) (the MXNet logger to TensorBoard).

![Figure 5: Predicted bounding boxes (dotted lines) compared with the actual bounding boxes (solid lines) as the network was trained to minimise the mean squared error (visualised with [MXBoard](https://github.com/awslabs/mxboard)).](https://cdn-images-1.medium.com/max/2150/1*wBBr0Zv0z2l5TXySwCuNaA.gif)*Figure 5: Predicted bounding boxes (dotted lines) compared with the actual bounding boxes (solid lines) as the network was trained to minimise the mean squared error (visualised with [MXBoard](https://github.com/awslabs/mxboard)).*

As the weights of the network were randomly initialised, we can observe that the predicted bounding boxes initially tended towards the bottom right corner of the image. We can observe that as number of iterations increased, the predicted bounding box drifted towards the correct area. During the last few epochs presented (240 & 280), the size of the bounding boxes fluctuated and the network most likely overfitted to the training data (see Figure 6).

![Figure 6: The loss curve of training to minimise the mean squared error (blue line — test loss, orange line — training loss)](https://cdn-images-1.medium.com/max/4078/1*3jyi73Q1trKdwJDUjpRhgQ.png)*Figure 6: The loss curve of training to minimise the mean squared error (blue line — test loss, orange line — training loss)*

The mean squared error was initially used as loss function because the intersection over union (IOU) loss function requires overlap between the predicted and actual bounding boxes (otherwise the values will be undefined). Therefore, after reasonable bounding boxes were generated with the mean squared area, the network was fine-tuned to minimise the IOU. The final results are shown in Figure 6.

![Figure 6: Final predicted bounding box (dotted lines) compared with the actual bounding boxes (solid lines) after the network was fine-tuned by minimising the IOU.](https://cdn-images-1.medium.com/max/2150/1*HEb82jJp93I0EFgYlJhfAw.png)*Figure 6: Final predicted bounding box (dotted lines) compared with the actual bounding boxes (solid lines) after the network was fine-tuned by minimising the IOU.*

### Data augmentation

Most writers start their passage in a similar position, so the predicted position of the bounding box was biased. To circumvent this issue, simple random translation was introduced. The training images were randomly shifted by 5% and the images are shown in Figure 7.

![Figure 7: Training images that were randomly translated (visualised with [MXBoard](https://github.com/awslabs/mxboard)).](https://cdn-images-1.medium.com/max/2150/1*ngt5RQjn6HehH81YVIkwFg.gif)*Figure 7: Training images that were randomly translated (visualised with [MXBoard](https://github.com/awslabs/mxboard)).*

Check out [this Jupyter Notebook](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/master/paragraph_segmentation_dcnn.ipynb) for an reference implementation of the deep CNN used above.

The deep CNN methods presented was able to identify the location of a handwritten passage and was able to generalise between different writers. The current implementation is limited to a single block of handwritten text in a relatively confined manner (e.g., one continuous block, no slanted writing, in a similar position). In the future, a more general page segmentation method will be developed which includes detecting multiple paragraphs and/or lines within paragraphs. The results of the page segmentation method will be fed into RNN based handwriting recognition algorithm.

The github repository can be found [here](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC).

## References

[1] Chen, K., Seuret, M., Liwicki, M., Hennebert, J., & Ingold, R. (2015, August). Page segmentation of historical document images with convolutional autoencoders. In *Document Analysis and Recognition (ICDAR), 2015 13th International Conference on* (pp. 1011–1015). IEEE.
[2] Puigcerver, J. (2017, November). Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?. In *Document Analysis and Recognition (ICDAR), 2017 14th IAPR International Conference on* (Vol. 1, pp. 67–72). IEEE.
[3] Bluche, T., Louradour, J., & Messina, R. (2017, November). Scan, attend and read: End-to-end handwritten paragraph recognition with mdlstm attention. In *Document Analysis and Recognition (ICDAR), 2017 14th IAPR International Conference on* (Vol. 1, pp. 1050–1055). IEEE.
[4] Marti, U. V., & Bunke, H. (2002). The IAM-database: an English sentence database for offline handwriting recognition. *International Journal on Document Analysis and Recognition*, *5*(1), 39–46.
[5] [https://www.mathworks.com/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html](https://www.mathworks.com/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html)
[6] Chen, H., Tsai, S. S., Schroth, G., Chen, D. M., Grzeszczuk, R., & Girod, B. (2011, September). Robust text detection in natural images with edge-enhanced maximally stable extremal regions. In *Image Processing (ICIP), 2011 18th IEEE International Conference on* (pp. 2609–2612). IEEE.

Images in Figure 1 were obtained from: [https://www.popsugar.com/moms/Kid-Accurate-Mother-Day-Card-44841930](https://www.popsugar.com/moms/Kid-Accurate-Mother-Day-Card-44841930), [https://www.bestfakedoctorsnotes.net/dr-generator/](https://www.bestfakedoctorsnotes.net/dr-generator/), [https://intellogist.wordpress.com/2011/12/13/free-online-sources-of-historical-scientific-journals-and-documents/](https://intellogist.wordpress.com/2011/12/13/free-online-sources-of-historical-scientific-journals-and-documents/)
