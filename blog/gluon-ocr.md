
# Handwriting OCR: Line segmentation with Gluon

Author: Jonathan Chung, Applied scientist intern at Amazon.

Following our previous [blog post](https://medium.com/apache-mxnet/page-segmentation-with-gluon-dcb4e5955e2), our pipeline to automatically recognize handwritten text includes: page segmentation and line segmentation, followed by handwriting recognition.

![Figure 1: Pipeline for handwriting recognition.](https://cdn-images-1.medium.com/max/12820/1*nJ-ePgwhOjOhFH3lJuSuFA.png)*Figure 1: Pipeline for handwriting recognition.*

In the previous [blog post](https://medium.com/apache-mxnet/page-segmentation-with-gluon-dcb4e5955e2), I described a deep convolutional neutral network (CNN) method to identify the location of the handwritten passage (page segmentation). In the current blog post, the passage of handwritten text is segmented line by line (line segmentation) so that each line can be used for handwriting recognition.

## Methods

The input to the model is an image that only contains handwritten text. The outputs are bounding boxes that correspond to each line of the text (see Figure 1). The problem is similar to the object detection problem in computer vision. In this method, we utilized the single shot multibox detector (SSD) architecture to detect the positions of each line of the passage.

The SSD architecture essentially takes image features and repeatedly downsamples the features (to account for different scaling factors). At each downsample step, the features are fed into two CNNs: one to estimate the locations of bounding boxes relative to anchor points, and one to estimate the probability of the bounding box encompassing an object. See [blog post](https://medium.com/@ManishChablani/ssd-single-shot-multibox-detector-explained-38533c27f75f) [1], [tutorial](https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html) [2], or the [original paper](https://arxiv.org/abs/1512.02325) [3] for more details of the SSD.

SSD was implemented using MXNet based on [tutorial](https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html) [2] and was optimized to this application by altering:

* Network architecture

* Anchor points

* Data augmentation

* Non-maximum suppression

### Network architecture

![Figure 2: Network architecture.](https://cdn-images-1.medium.com/max/9646/1*jMkO7hy-1f0ZFHT3S2iH0Q.png)*Figure 2: Network architecture.*

The main differences between the current network architecture and the one described in the [tutorial](https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html) [2] is the use of Resnet 34 to extract image features. The first convolutional layer of a pre-trained Resnet 34 (RGB) was replaced with a 1-channel convolutional layer (grayscale) by averaging the weights of the RGB channels.

### Anchor points

The bounding boxes encompassing lines of handwritten text are mostly restricted to horizontal rectangles. On the other hand, the bounding boxes required for general object detection dramatically vary in size (Figure 3-a). Therefore, rectangles with aspect ratios >1 were chosen for the current application and is shown in Figure 3-b. We also utilized two more anchor points compared to the blog post.

![Figure 3: a) Anchor boxes used in the tutorial [2]. b) Anchor boxes used in the current application.](https://cdn-images-1.medium.com/max/3248/1*j1Hto2P7OGnUBhyfuSeEKQ.png)*Figure 3: a) Anchor boxes used in the tutorial [2]. b) Anchor boxes used in the current application.*

### Data augmentation

The paper [3] emphasized the importance of data augmentation when training the SSD model (the [tutorial](https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html) [2] did not include data augmentation). The authors used random translations, cropping, and flipping however for the current application, random cropping and flipping is not appropriate as cropping will compromise the continuity of the text and flipping will reverse the writing direction. In this work, we similarly used random translations and we also introduced a method that randomly removed lines (that may be similar to random cropping). Specifically, each line was removed with a probability of *p (p=0.15* in this work*). *The image was filled with the color of the background where the bounding box is located and then the bounding box was removed (see Figure 4).

![Figure 4: Example of data augmentation by randomly replacing lines with the background color of the document. Dotted lines are the predicted bounding boxes and solid lines are the labelled bounding boxes.](https://cdn-images-1.medium.com/max/2000/1*0xhjxqYYUHg6WvFjCYh0Lg.png)*Figure 4: Example of data augmentation by randomly replacing lines with the background color of the document. Dotted lines are the predicted bounding boxes and solid lines are the labelled bounding boxes.*

### Non-maximum suppression

The network predicts numerous overlapping and redundant bounding boxes (as shown in Figure 5-a). To obtain more meaningful results, the box_nms (box non-maximum suppression) [4] function was applied on the output of the network. Three parameters: overlap threshold (overlap_thres), top k boxes (topk), and minimum threshold (valid_thres) were varied and tuned. The parameters overlap_thres=0.1, topk=150, and valid_thres=0.01 were selected and the results are shown in Figure 5-b (note that the results shown in Figure 4 were passed through the non-maximum suppression algorithm).

![Figure 5: Demonstration of the non-maximum suppression algorithm. a) with no non-maximum suppression algorithm, b) with non-maximum suppression algorithm (overlap_thres=0.1, topk=150, and valid_thres=0.01).](https://cdn-images-1.medium.com/max/2000/1*aOSkuCm7CMP1HkfR1pKfLw.png)*Figure 5: Demonstration of the non-maximum suppression algorithm. a) with no non-maximum suppression algorithm, b) with non-maximum suppression algorithm (overlap_thres=0.1, topk=150, and valid_thres=0.01).*

## Results

The final results are shown in Figure 6. As shown in Figure 6-a, we can see that the predicted bounding boxes have large overlaps with the labelled bounding boxes (train IOU = 0.593, test IOU = 0.573). We also observed that the network can learn the bounding boxes for handwriting with distinct lines substantially easier than large handwriting which overlaps. Several interesting examples are also presented in Figure 6-b and c.

![Figure 6: a) Predicted bounding boxes from the network, b) Example where an incorrectly labelled line was predicted correctly, c) Example where the predicted box was misaligned (dotted lines are predicted bounding boxes, solid lines are labelled bounding boxes).](https://cdn-images-1.medium.com/max/2118/1*JJGwLXJL-bV7zsfrfw84ew.png)*Figure 6: a) Predicted bounding boxes from the network, b) Example where an incorrectly labelled line was predicted correctly, c) Example where the predicted box was misaligned (dotted lines are predicted bounding boxes, solid lines are labelled bounding boxes).*

### Training progression

In Figure 7, the training and testing loss, and the mean absolute error (L1 loss) is presented as a function of the epochs. The predicted bounding boxes are also predicted as the epochs progressed, this is presented in Figure 8.

![Figure 7: a) Loss of the network (location + classification) as a function of epochs (blue line is the training loss, orange line is the testing loss), b) Testing mean absolute error of the predicted bounding boxes.](https://cdn-images-1.medium.com/max/2000/1*I-9wP6uBimyWANy7NLJLBg.png)*Figure 7: a) Loss of the network (location + classification) as a function of epochs (blue line is the training loss, orange line is the testing loss), b) Testing mean absolute error of the predicted bounding boxes.*

![Figure 8: Predicted bounding boxes as the network is trained (dotted lines are predicted bounding boxes, solid lines are labelled bounding boxes).](https://cdn-images-1.medium.com/max/2118/1*XiXx1ZnRe_es7rvOYhC-ig.gif)*Figure 8: Predicted bounding boxes as the network is trained (dotted lines are predicted bounding boxes, solid lines are labelled bounding boxes).*

## Limitations

The current algorithm is limited to images that only contain handwritten text, with no images or printed text. Also the text must be confined in consecutively written horizontal lines. As such, this algorithm will have difficulties predicting the bounding boxes of handwritten notes with more varied layouts.

Check out [this Jupyter Notebook](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/master/line_segmentation.ipynb) for a reference implementation of the SSD algorithm used above.

## References

[1] [https://medium.com/@ManishChablani/ssd-single-shot-multibox-detector-explained-38533c27f75f](https://medium.com/@ManishChablani/ssd-single-shot-multibox-detector-explained-38533c27f75f)

[2] [https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html](https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html)

[3] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016, October). Ssd: Single shot multibox detector. In *European conference on computer vision* (pp. 21–37). Springer, Cham.

[4] [https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html#mxnet.ndarray.contrib.box_nms](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html#mxnet.ndarray.contrib.box_nms)
