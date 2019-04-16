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
      :title: Tuning NumPy Operations
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
      :link: compression/float16.html

      How to use float16 in your model to boost training speed.

   .. card::
      :title: Gradient Compression
      :link: compression/gradient_compression.html

      How to use gradient compression to reduce communication bandwidth and increase speed.

   ..
      TBD Content
      .. card::
         :title: Compression: int8
         :link: compression/int8.html

         How to use int8 in your model to boost training speed.
   ..

Accelerated Backend
-------------------

.. container:: cards

   .. card::
      :title: TensorRT
      :link: backend/tensorRt.html

      How to use NVIDIA's TensorRT to boost inference performance.

   ..
      TBD Content
      .. card::
         :title: MKL-DNN
         :link: backend/mkl-dnn.html

         How to get the most from your CPU by using Intel's MKL-DNN.

      .. card::
         :title: TVM
         :link: backend/tvm.html

         How to use TVM to boost performance.
   ..

.. toctree::
   :hidden:
   :maxdepth: 1

   compression/index
   backend/index
