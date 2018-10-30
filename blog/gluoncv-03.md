
# GluonCV 0.3: A New Horizon

Author: Zhi Zhang Amazon AI Applied Scientist

Translated from: [https://zh.mxnet.io/blog/gluon-cv-0.3](https://zh.mxnet.io/blog/gluon-cv-0.3)

Translator: Hao Jin Amazon AI Software Development Engineer

We started the GluonCV project about half a year ago, looking to provide a reliable deep learning computer vision library that reproduces results of various research papers. For the past few months we dived deeply into numerous papers and corresponding implementations to discover the hidden details in them, and we conducted many experiments using those details. We were excited to find that, for some of the papers, we could do even better than merely reproducing their results.

Now we are glad to release Gluon CV v0.3. In this new version we added 5 new algorithms and 38 new pre-trained models, together with major improvements to 28 models from v0.2. The following table compares 7 of them:

![](https://cdn-images-1.medium.com/max/2000/1*FlYKsapwq-7MzZ7Nn1s-xQ.png)

Something worth mentioning is that we greatly improved the accuracies of the ResNet family. The ResNet-50 trained by us gives 1% more accuracy at the cost of only 1/3 computation compared to the ResNet-152 model from the original paper. We also improved the mAP accuracy of YOLOv3 by 4%, which is about the same as Faster R-CNN with ResNet-50, but with 10x improvement to both speed and memory efficiency.

At the same time, we improved our Model Zoo page user experience so that, now, one can interactively cross-compare the accuracy and performance values of all models.

What’s more, we also improved the deployability of models. Every single model can be run without using Python, that is, all of them can now be hybridized.

Now let’s dive into the details of the improvements in v0.3.

## Research and Competitions

GluonCV is definitely a good weapon for achieving high ranks in competitions. Even with the default parameters, one could readily rank top on various open datasets. Furthermore, our modular design makes it easy to integrate various new ideas without sacrificing much time and effort, so that deadlines can be met comfortably. v0.3 also comes with several tutorials on preparing your customized dataset to enable fast environment preparations. We’ll keep adding more tutorials to make everyone’s experience with GluonCV more comfortable.

Click on the picture below to see accuracies of the pre-trained image classification models.

![](https://cdn-images-1.medium.com/max/2000/1*f9h5BVlZXp6G_n9UkuuOCA.png)

Click the picture below to see accuracies of the pre-trained object detection models.

![](https://cdn-images-1.medium.com/max/2000/1*Qqw49OzdsmLAP99MFIVRRQ.png)

## What’s in v0.3

**New Model: [YOLOv3](https://gluon-cv.mxnet.io/model_zoo/detection.html#ms-coco)**

YOLOv3 is the version with the most improvements since YOLOv1. We made slight modifications on top of the original paper and improved mAP of YOLOv3 on COCO dataset from 33% to 37% (at 608 input resolution), which means parallel accuracy of Faster-RCNN with ResNet-50 backbone with an extra >10x speedup

**New Model: [Mask-RCNN](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#instance-segmentation)**

Mask-RCNN is a multi-task model based on Faster-RCNN family, which added semantic segmentation feature in addition to object detection. v0.3 provides a pre-trained model based on ResNet-50 backbone, which delivers 38.3% mAP for object detection and 33.1% mAP for semantic segmentation. These numbers are slightly better than equivalent model from Detectron.

**New Model: [DeepLabV3](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation)**

DeepLabV3 is an end-to-end model based on FCN. It uses multi-scale context to improve the accuracy of semantic segmentation, and delivers magnificent performance on Pascal VOC and ADE20K datasets. Our reproduction of the model exceeds the original paper on VOC dataset with a high accuracy of 86.7 mIoU.

**Performance Improvement: [Faster-RCNN](https://gluon-cv.mxnet.io/model_zoo/detection.html#ms-coco)**

The highlight of our Faster-RCNN optimizations is that we succeeded in achieving 40.1% mAP on COCO dataset based on vanilla ResNet-101 without FPN, which exceeds the 39.6% mAP accuracy by Detectron with ResNet-101 backbone. The model based on ResNet-50 also achieved a 37% score, which is the best among peer open-source implementations. We’ll add FPN support in the upcoming versions. More details on the tricks we used for accuracy improvements will be included in our new paper.

**Performance Improvement: [Image Classification](https://gluon-cv.mxnet.io/model_zoo/classification.html)**

Exceeding 79% accuracy with ResNet-50? Getting more than 80% accuracy with ResNet-101? Those are no longer your wildest dreams! With our recent work on minor changes to model structures and extra optimizations for the training process, we further enhanced accuracies of ResNet models to today’s highest without additional computation or model complexity. In addition to the ResNet family, we have also boosted accuracies of MobileNet family, DarkNet53, and Inception V3 with the same optimizations and led to better results than original papers. Meanwhile, the improved models have also achieved better results on other tasks such as object detection and semantic segmentation, which means that such improvements are not merely overfitting for ImageNet dataset.

For more details, please refer to GluonCV’s [Model Zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html), additional related details could also be found in our upcoming paper.

**New Application: [GAN](https://github.com/dmlc/gluon-cv/tree/master/scripts/gan/wgan)**

v0.3 has added popular GAN applications, [@husonchen](https://github.com/husonchen) contributed WGAN model and corresponding training scripts. The following picture of bedrooms shows samples of images generated by WGAN.

![](https://cdn-images-1.medium.com/max/2000/0*hZW5mO_WsvNVxkwZ.png)

**New Application:[Person Re-identification](https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/baseline)**

Person Re-identification is a very important application in the field of security. In v0.3, [@xiaolai-sqlai](https://github.com/xiaolai-sqlai) contributed the training module for person re-identification, which got a 93.1% best result on market1501 dataset.

**Performance Improvement: [Semantic Segmentation](https://gluon-cv.mxnet.io/model_zoo/segmentation.html)**

With [multi-GPU Synchronized Batchnorm](https://zh.mxnet.io/blog/syncbn) added to MXNet, we now provide complete training code based on SynchBN for perfect reproduction of SOTA semantic segmentation algorithms. Plus, we also added CityScapes dataset and pre-trained PSPNet model. Since the reproduction steps on Pascal VOC dataset are quite complicated, we provide a [detailed tutorial](https://gluon-cv.mxnet.io/build/examples_segmentation/voc_sota.html) to walk you through the reproduction step-by-step. With GluonCV, one can not only reproduce the SOTA results, but also learn more about the details.

**New Feature: [Deployment](https://gluon-cv.mxnet.io/build/examples_deployment/export_network.html)**

Need to deploy your trained models? GluonCV has got your back. GluonCV provides an one-click export function, together with example inference code in C++, which will speed up your deployment effort. Also, if the pre-trained model already satisfies your needs, simply skip the training and deploy directly!

## **Next**

- Link to GluonCV: [https://gluon-cv.mxnet.io/index.html](https://gluon-cv.mxnet.io/index.html](https://gluon-cv.mxnet.io/index.html))

- GluonCV GitHub：[https://github.com/dmlc/gluon-cv](https://github.com/dmlc/gluon-cv](https://github.com/dmlc/gluon-cv))

- Gluon Forum: [https://discuss.mxnet.io/](https://discuss.mxnet.io/](https://discuss.mxnet.io/))

Please remember to [star](https://github.com/dmlc/gluon-cv) if you like GluonCV!
