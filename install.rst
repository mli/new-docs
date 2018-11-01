Install
=======

.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option
.. role:: dis
   :class: disable option

.. container:: install

   .. container:: opt-group

      :title:`OS:`
      :opt:`Linux`
      :opt:`Macos`
      :opt:`Windows`

   .. container:: opt-group

      :title:`Package:`
      :act:`Pip`
      :opt:`Docker`
      :opt:`Source`

   .. container:: pip docker opt-group

      :title:`Backend:`
      :act:`Native`
      :opt:`CUDA`
      :opt:`MKL-DNN`
      :opt:`CUDA + MKL-DNN`

   .. container:: pip docker

      .. admonition:: Prerequisite:

         .. container:: docker

            - Assume `docker <https://docs.docker.com/install/>`_ is installed and
              can be used by a non-root user.

         .. container:: docker

              .. container:: cuda cuda-mkl-dnn

                 - You need to install `nvidia-docker
                   <https://github.com/NVIDIA/nvidia-docker>`_

         .. container:: pip

            - A recent `pip <https://pip.pypa.io/en/stable/installing/>`_
              >= 9 is required. Please refer to `Issue 8671
              <https://github.com/apache/incubator-mxnet/issues/8671>`_ for all
              variants and compiliations flags.

         .. container:: cuda cuda-mkl-dnn

            - You need to install the according version CUDA to run on Nvidia
              GPUs. All CUDA version can be found `here.
              <https://developer.nvidia.com/cuda-toolkit-archive>`_

         .. container:: mkl-dnn cuda-mkl-dnn

            - The MKL-DNN variant provides accelerated performance on Intel CPUs. It
              can be combined with CUDA as well, such as ``mxnet-cu92mkl``.

      .. admonition:: Command:

         .. container:: pip

            .. container:: native

               .. code-block:: bash

                  pip install mxnet

            .. container:: cuda

               .. code-block:: bash

                  # Assume you installed CUDA 9.2, you may change the number
                  # according to your own CUDA version
                  pip install mxnet-cu92

            .. container:: mkl-dnn

               .. code-block:: bash

                  pip install mxnet-mkl

            .. container:: cuda-mkl-dnn

               .. code-block:: bash

                  # Assume you installed CUDA 9.2, you may change the number
                  # according to your own CUDA version
                  pip install mxnet-cu92mkl

         .. container:: docker

            .. container:: native

               .. code-block:: bash

                  docker pull mxnet/python

            .. container:: cuda

               .. code-block:: bash

                  docker pull mxnet/python:gpu

            .. container:: mkl-dnn

               .. code-block:: bash

                  docker pull mxnet/python:1.3.0_cpu_mkl

            .. container:: cuda-mkl-dnn

               .. code-block:: bash

                   docker pull mxnet/python:1.3.0_gpu_cu90_mkl_py3

   .. container:: source

      .. admonition:: Instruction:

         Follow instructions at this URL: xxx

.. raw:: html

   <script type="text/javascript" src='_static/install-options.js'></script>
