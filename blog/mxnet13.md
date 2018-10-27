
# Announcing Apache MXNet 1.3

Authors: Roshani Nagmote, Sheng Zha

Today the Apache MXNet community is pleased to announce the 1.3 release of the Apache MXNet deep learning framework. We would like to thank the Apache MXNet community for all their valuable contributions towards the MXNet 1.3 release.

With this release, MXNet has Gluon package enhancements, ONNX export, experimental Clojure bindings, TensorRT integration, and many more features, enhancements and usability improvements! In this blog post, we briefly summarize some of the high-level features and improvements. For a comprehensive list of major features and bug fixes, read the Apache [MXNet 1.3.0 release notes](https://github.com/apache/incubator-mxnet/releases).

![](https://cdn-images-1.medium.com/max/2400/1*P8ILqYgfsUYOPuDz3AUhWg.png)

## Gluon package enhancements

**Gluon RNN layers are now hybridizable: **With this feature, Gluon RNN layers such as [gluon.rnn.RNN](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.RNN), [gluon.rnn.LSTM](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.LSTM)and [gluon.rnn.GRU](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.GRU) can be converted to [HybridBlocks](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=hybri#mxnet.gluon.HybridBlock). Now, many dynamic networks that are based on Gluon RNN layers can be completely hybridized, exported and used in the inference APIs in other language bindings such as C/C++, Scala, R, etc.

**Support for sparse tensor: **Gluon** [**HybridBlocks](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.HybridBlock) now support hybridization with sparse operators. To enable sparse gradients in [gluon.nn.Embedding](https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Embedding), simply set sparse_grad=True. Furthermore, [gluon.contrib.nn.SparseEmbedding](https://mxnet.incubator.apache.org/api/python/gluon/contrib.html#mxnet.gluon.contrib.nn.SparseEmbedding) provides an example of leveraging sparse parameters to reduce communication cost and memory consumption for multi-GPU training with large embeddings.

**Support for Synchronized Cross-GPU Batch Norm: **Gluon now supports [Synchronized Batch Normalization](https://mxnet.incubator.apache.org/api/python/gluon/contrib.html#mxnet.gluon.contrib.nn.SyncBatchNorm). This enables stable training on large-scale networks with high memory consumption such as FCN for image segmentation.

**Updated Gluon model zoo:** [Gluon Vision Model Zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html) now provides MobileNetV2 pre-trained models. Updated existing pre-trained models to provide state-of-the-art performance on all ResNet v1, ResNet v2, and vgg16, vgg19, vgg16_bn, vgg19_bn models.

## **Introducing new Clojure bindings with MXNet**

MXNet now has experimental support for the Clojure programming language. The MXNet Clojure package brings state-of-the-art deep learning to the Clojure community. It enables Clojure developers to code and to execute tensor computation on multiple CPUs or GPUs. It also enables users to write seamless tensor/matrix computations with multiple GPUs in Clojure. Now users can construct and customize state-of-art deep learning models in Clojure, and apply them to tasks such as image classification and data science challenges. To start using Clojure package in MXNet, check out the [Clojure tutorials](http://mxnet.incubator.apache.org/api/clojure/index.html#clojure-api-tutorials) and [Clojure API documentation](http://mxnet.incubator.apache.org/api/clojure/index.html).

## **Introducing control flow operators**

This is the first step towards optimizing dynamic neural networks with variable computation graphs. This release adds symbolic and imperative control flow operators such as [foreach](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html#mxnet.ndarray.contrib.foreach), [while_loop](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html#mxnet.ndarray.contrib.while_loop) and [cond](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html#mxnet.ndarray.contrib.cond). To learn more about how to use these operators, check out the [Control Flow Operators tutorial](https://mxnet.incubator.apache.org/tutorials/control_flow/ControlFlowTutorial.html).

## Performance improvements

**TensorRT runtime integration: [**TensorRT](https://developer.nvidia.com/tensorrt) provides significant acceleration of model inference on NVIDIA GPUs compared to running the full graph in MXNet using unfused GPU operators. In addition to faster fp32 inference, TensorRT optimizes fp16 inference and is capable of int8 inference (provided the quantization steps are performed). Besides increasing throughput, TensorRT significantly reduces inference latency, especially for small batches. With 1.3 release, MXNet introduces the [runtime integration of TensorRT](http://mxnet.incubator.apache.org/tutorials/tensorrt/inference_with_trt.html) (experimental), in order to accelerate inference. Follow the [MXNet-TensorRT article](https://cwiki.apache.org/confluence/display/MXNET/How+to+use+MXNet-TensorRT+integration) on the [MXNet developer wiki](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Home) to learn more about how to use this feature.

**MKL-DNN enhancements: [**MKL-DNN](https://01.org/mkl-dnn) is an open source library from Intel that contains a set of CPU-optimized deep learning operators. In the previous release, MXNet introduced integration with MKL-DNN to accelerate training and inference execution on CPU. With 1.3 release, we have increased support for these activation functions: sigmoid, tanh and softrelu.

## ONNX export support

**Export MXNet models to ONNX format:** MXNet 1.2 provided users a way to import ONNX models into MXNet for inference. More details are available in [this ONNX blog post](https://medium.com/apache-mxnet/mxnet-1-2-adds-built-in-support-for-onnx-e2c7450ffc28). With the latest 1.3 release, users can now export MXNet models into ONNX format and import those models into other deep learning frameworks for inference! Check out the [MXNet to ONNX exporter tutorial](http://mxnet.incubator.apache.org/tutorials/onnx/export_mxnet_to_onnx.html) to learn more about how to use the [mxnet.contrib.onnx API](http://mxnet.incubator.apache.org/api/python/contrib/onnx.html).

## Other experimental features

Apart from what we have covered above, MXNet now has support for:

1. A new memory pool type for GPU memory which is more suitable for all the workloads with dynamic-shape inputs and outputs. Set an environment variable asMXNET_GPU_MEM_POOL_TYPE=Round to enable this feature.

1. [Topology-aware Allreduce](https://cwiki.apache.org/confluence/display/MXNET/Single+machine+All+Reduce+Topology-aware+Communication) approach for single-machine GPU training. Train up to [6.6x and 5.9x faster](https://cwiki.apache.org/confluence/display/MXNET/Single+machine+All+Reduce+Topology-aware+Communication#SinglemachineAllReduceTopology-awareCommunication-End-to-EndResults) on AlexNet and VGG compared to MXNet 1.2. Activate this feature using the [“control the data communication” environmental variables](https://mxnet.incubator.apache.org/faq/env_var.html#control-the-data-communication).

1. [Improved Scala APIs](https://cwiki.apache.org/confluence/display/MXNET/Scala+Type-safe+API+Design+Doc) that focus on providing type safety and a better user experience. [Symbol.api](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.SymbolAPI$) and [NDArray.api](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.NDArrayAPI$) bring a new set of functions that have a complete signature. The documentation for all of the arguments also integrates directly with IntelliJ IDEA. The new and improved [Scala examples](https://github.com/apache/incubator-mxnet/tree/v1.3.x/scala-package/examples/src/main/scala/org/apache/mxnetexamples) demonstrate usage of these new APIs.

Check out further details on these features in full [release notes](https://github.com/apache/incubator-mxnet/releases).

## Maintenance improvements

In addition to adding and extending new functionalities, the release also focusses on stability and refinements.

1. The community fixed 130 [unstable tests](https://github.com/apache/incubator-mxnet/projects/9) improving MXNet’s stability and reliability.

1. The [MXNet Model Backwards Compatibility Checker](https://github.com/apache/incubator-mxnet/pull/11626) was introduced. This is an automated test on MXNet’s continuous integration platform that verifies saved models’ backward compatibility. This helps ensure that models created with older versions of MXNet can be loaded and used with the newer versions.

## Getting started with MXNet

Getting started with MXNet is simple, visit the [install page](https://mxnet.incubator.apache.org/install/index.html) to get started. [Pypi packages](https://pypi.org/project/mxnet/) are available to install for Linux, Mac, and Windows!

To learn more about MXNet Gluon package and deep learning, you can follow our [60-minute crash course](https://medium.com/apache-mxnet/mxnet-gluon-in-60-minutes-3d49eccaf266), and then later complete this [comprehensive set of tutorials](http://gluon.mxnet.io/), which covers everything from an introduction to deep learning to how to implement cutting-edge neural network models. You can also check out lots of material on [MXNet tutorials](https://mxnet.incubator.apache.org/tutorials/index.html), [MXNet blog posts](https://medium.com/apache-mxnet), and [MXNet YouTube channel](https://www.youtube.com/channel/UCQua2ZAkbr_Shsgfk1LCy6A). Have fun with MXNet 1.3.0!

## Acknowledgments:

We would like to thank everyone who contributed to the 1.3.0 release:

Aaron Markham, Abhinav Sharma, access2rohit, Alex Li, Alexander Alexandrov, Alexander Zai, Amol Lele, Andrew Ayres, Anirudh Acharya, Anirudh Subramanian, Ankit Khedia, Anton Chernov, aplikaplik, Arunkumar V Ramanan, Asmus Hetzel, Aston Zhang, bl0, Ben Kamphaus, brli, Burin Choomnuan, Burness Duan, Caenorst, Cliff Woolley, Carin Meier, cclauss, Carl Tsai, Chance Bair, chinakook, Chudong Tian, ciyong, ctcyang, Da Zheng, Dang Trung Kien, Deokjae Lee, Dick Carter, Didier A., Eric Junyuan Xie, Faldict, Felix Hieber, Francisco Facioni, Frank Liu, Gnanesh, Hagay Lupesko, Haibin Lin, Hang Zhang, Hao Jin, Hao Li, Haozhi Qi, hasanmua, Hu Shiwen, Huilin Qu, Indhu Bharathi, Istvan Fehervari, JackieWu, Jake Lee, James MacGlashan, jeremiedb, Jerry Zhang, Jian Guo, Jin Huang, jimdunn, Jingbei Li, Jun Wu, Kalyanee Chendke, Kellen Sunderland, Kovas Boguta, kpmurali, Kurman Karabukaev, Lai Wei, Leonard Lausen, luobao-intel, Junru Shao, Lianmin Zheng, Lin Yuan, lufenamazon, Marco de Abreu, Marek Kolodziej, Manu Seth, Matthew Brookhart, Milan Desai, Mingkun Huang, miteshyh, Mu Li, Nan Zhu, Naveen Swamy, Nehal J Wani, PatricZhao, Paul Stadig, Pedro Larroy, perdasilva, Philip Hyunsu Cho, Pishen Tsai, Piyush Ghai, Pracheer Gupta, Przemyslaw Tredak, Qiang Kou, Qing Lan, qiuhan, Rahul Huilgol, Rakesh Vasudevan, Ray Zhang, Robert Stone, Roshani Nagmote, Sam Skalicky, Sandeep Krishnamurthy, Sebastian Bodenstein, Sergey Kolychev, Sergey Sokolov, Sheng Zha, Shen Zhu, Sheng-Ying, Shuai Zheng, slitsey, Simon, Sina Afrooze, Soji Adeshina, solin319, Soonhwan-Kwon, starimpact, Steffen Rochel, Taliesin Beynon, Tao Lv, Thom Lane, Thomas Delteil, Tianqi Chen, Todd Sundsted, Tong He, Vandana Kannan, vdantu, Vishaal Kapoor, wangzhe, xcgoner, Wei Wu, Wen-Yang Chu, Xingjian Shi, Xinyu Chen, yifeim, Yizhi Liu, YouRancestor, Yuelin Zhang, Yu-Xiang Wang, Yuan Tang, Yuntao Chen, Zach Kimberg, Zhennan Qin, Zhi Zhang, zhiyuan-huang, Ziyue Huang, Ziyi Mu, Zhuo Zhang.

… and thanks to all of the Apache MXNet community supporters, spreading knowledge and helping to grow the community!
