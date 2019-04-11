Performance
===========
The following tutorials will help you learn how to tune MXNet or use tools that will improve training and inference performance.

Essential
---------

.. container:: cards

   .. card::
      :title: Improving Performance
      :link: perf.html

      How to get the best performance from MXNet.

   .. card::
      :title: Profiler
      :link: profiler.html

      How to profile MXNet models.

   .. card::
      :title: Tuning Numpy Operations
      :link: numpy.html

      Gotchas using NumPy in MXNet.

.. toctree::
   :hidden:
   :maxdepth: 1

   perf
   profiler
   numpy


Compression
-----------

.. container:: cards

   .. card::
      :title: Compression: float16
      :link: float16.html

      How to use float16 in your model to boost training speed.

   .. card::
      :title: Compression: int8
      :link: index.html

      How to use int8 in your model to boost training speed.

   .. card::
      :title: Gradient Compression
      :link: gradient_compression.html

      How to use gradient compression to reduce communication bandwidth and increase speed.

.. toctree::
   :hidden:
   :maxdepth: 1

   float16
   int8
   gradient_compression

Accelerated Backend
-------------------

.. container:: cards

   .. card::
      :title: MKL-DNN
      :link: mkl-dnn.html

      How to get the most from your CPU by using Intel's MKL-DNN.

   .. card::
      :title: TensorRT
      :link: index.html

      How to use NVIDIA's TensorRT to boost inference performance.

   .. card::
      :title: TVM
      :link: tvm.html

      How to use TVM to boost performance.


.. toctree::
   :hidden:
   :maxdepth: 1

   mkl-dnn
   tensorRt
   tvm
