Performance
===========
The following tutorials will help you learn how to tune MXNet or use tools that will improve training and inference performance.

.. container:: cards

   .. card::
      :title: Improving Performance
      :link: performance/perf.html

      How to get the best performance from MXNet.

   .. card::
      :title: Profiler
      :link: performance/profiler.html

      How to profile MXNet models.

   .. card::
      :title: Tuning Numpy Operations
      :link: performance/numpy.html

      Gotchas using NumPy in MXNet.

   .. card::
      :title: Compression: float16
      :link: performance/float16.html

      How to use float16 in your model to boost training speed.

   .. card::
      :title: Compression: int8
      :link: performance/index.html

      How to use int8 in your model to boost training speed.

   .. card::
      :title: Gradient Compression
      :link: performance/gradient_compression.html

      How to use gradient compression to reduce communication bandwidth and increase speed.

   .. card::
      :title: MKL-DNN
      :link: performance/mkl-dnn.html

      How to get the most from your CPU by using Intel's MKL-DNN.

   .. card::
      :title: TensorRT
      :link: performance/index.html

      How to use NVIDIA's TensorRT to boost inference performance.

   .. card::
      :title: TVM
      :link: performance/tvm.html

      How to use TVM to boost performance.


Essential
---------

.. toctree::
   :hidden:
   :maxdepth: 1

   perf
   profiler
   numpy

Compression
-----------

.. toctree::
   :hidden:
   :maxdepth: 1

   float16
   int8
   gradient_compression

Accelerated Backend
-------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   mkl-dnn
   tensorRt
   tvm
