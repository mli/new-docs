
# Object Detection with MXNet Scala Inference API

Author: Qing Lan

![](https://cdn-images-1.medium.com/max/3412/1*bzQvLuHqgDv0s6QUbSYNaw.png)

With the recent release of MXNet version 1.2.0, the new [MXNet Scala Inference API](https://mxnet.incubator.apache.org/api/scala/infer.html) was released. This release focuses on optimizing the developer experience for inference applications written in Scala. Scala is a general-purpose programming language that supports both functional programming and a strong static type system, and is used with high scale distributed processing with platforms such as Apache Spark.

We recommend you check out the other posts in our MXNet Scala Inference API series, if you haven’t already seen them:

* [Scala API for Deep Learning Inference Now Available with MXNet v1.2](https://medium.com/apache-mxnet/scala-api-for-deep-learning-inference-now-available-with-mxnet-v1-2-bcb13235db95)

* [Image Classification with MXNet Scala Inference API](https://medium.com/apache-mxnet/image-classification-with-mxnet-scala-inference-api-8ab6ce1bbccf)

And in this last post, we work through an example of using the MXNet Scala Inference API for [**Object Detection](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector)**. This model is used to identify objects and their locations in an image. It is also known as a Single-Shot Detector (SSD) model. You can learn more about Object Detection [here](https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html).

Similar to image classification example in our previous post, you need to define the following paths. You can download our sample model using script [here](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/scripts/infer/objectdetector/get_ssd_data.sh).

    val modelPathPrefix = "/absolute/path/to/your/model"
    val inputImagePath = "/absolute/path/to/your/image"
    

Then prepare the setup of the model.

    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)
    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))
    val topK = Some(3) # Number of results
    val context = Context.cpu()

After this initial setup, you load the image with ImageClassifier and it will be pre-processed for you.

    val img = ImageClassifier.loadImageFromFile(inputImagePath)

Then you initialize a new ObjectDetector using the model location, its description, and execution context.

    val objectDetector = new ObjectDetector(modelPathPrefix, inputDescriptors, context)

After this step, you can use ObjectDetector for inference on the input image. In this example, you will get three results.

    val output = objectDetector.imageObjectDetect(img, topK)

The default output is a bit hard to read, so the example provides [this code](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector/SSDClassifierExample.scala#L136-L147) to help clean it up. If you use the following image, you will see a similar output as shown after the image.

![](https://cdn-images-1.medium.com/max/2000/1*5jqqcfUiT7cBiYlDJunlEw.jpeg)

    Class: car
    Probabilties: 0.98847263
    Coord:312.21335,72.0291,456.01443,150.66176
    Class: bicycle
    Probabilties: 0.94833825
    Coord:155.95807,149.96362,383.8369,418.94513
    Class: dog
    Probabilties: 0.8281818
    Coord:83.82353,179.13998,206.63783,476.7875

This is the generated image based on the result:

![](https://cdn-images-1.medium.com/max/2000/1*wt5dIndKyFAcklbZlY2tDw.png)

## Call for Contribution

If you’re a Scala user, and you like where this is going, [join the project](https://mxnet.incubator.apache.org/community/contribute.html), provide feedback, or pitch in on a feature you want to see. As an open source project, these great features are free to use, and are influenced and improved by open source community’s involvement. We are actively developing the training features with Scala with new Type-safe APIs and creating more examples using Scala on MXNet. Also, make sure you follow Apache MXNet to kept posted on new features.
