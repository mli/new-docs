.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

   .. container:: opt-group

      :title:`Platform:`
      :act:`Local`
      :opt:`Cloud`

   .. container:: cloud opt-group

      :title:`Provider:`
      :act:`AWS`

      .. container:: aws

         .. admonition:: Instruction:

            There are several options you can access MXNet at AWS:

            - `Deep Learning AMI
              <https://aws.amazon.com/machine-learning/amis/>`_: Amazon machine
              images with MXNet pre-installed, available for Ubuntu, Amazon
              Linux, and Windows 2016.
            - `Sagemaer <https://aws.amazon.com/sagemaker/>`_: a fully-managed
              machine learning platform with MXNet integrated.

   .. container:: local

      .. container:: opt-group

         :title:`OS:`
         :opt:`Linux`
         :opt:`Macos`
         :opt:`Windows`

      .. container:: opt-group

         :title:`Package:`
         :act:`Pip`
         :opt:`Docker`


      .. container:: opt-group

         :title:`Backend:`
         :act:`Native`
         :opt:`CUDA`
         :opt:`MKL-DNN`
         :opt:`CUDA + MKL-DNN`

         .. raw:: html

            <div class="mdl-tooltip" data-mdl-for="native">Build-in backend for CPU.</div>
            <div class="mdl-tooltip" data-mdl-for="cuda">Required to run on Nvidia GPUs.</div>
            <div class="mdl-tooltip" data-mdl-for="mkl-dnn">Accelarate Intel CPU performacne.</div>
            <div class="mdl-tooltip" data-mdl-for="cuda-mkl-dnn">Enable both Nvidia CPUs and Inter CPU accelaration.</div>

      .. admonition:: Prerequisite:

         .. container:: docker

            - Require `docker <https://docs.docker.com/install/>`_ is installed
              and it can be used by a non-root user.

         .. container:: docker

              .. container:: cuda cuda-mkl-dnn

                 - `nvidia-docker
                   <https://github.com/NVIDIA/nvidia-docker>`_ is required to
                   run on Nvidia GPUs.

         .. container:: pip

            - Require `pip >= 9. <https://pip.pypa.io/en/stable/installing/>`_ is
              installed. Both Python 2 and Python 3 are supported.
            - Hint: append the flag ``--pre`` at the end of the command will
              install the nightly build.
            .. - Hint: refer to `Issue 8671
               <https://github.com/apache/incubator-mxnet/issues/8671>`_ for
               all MXNet variants that available for pip.

            .. container:: cuda cuda-mkl-dnn

               - Require `CUDA
                 <https://developer.nvidia.com/cuda-toolkit-archive>`_ is
                 installed. Supported versions include 8.0, 9.0, and 9.2.
               - Hint: `cuDNN <https://developer.nvidia.com/cudnn>`_ is already
                 included in the MXNet binary, you don't need to install it.

            .. container:: mkl-dnn cuda-mkl-dnn

               - Hint: `MKL-DNN <https://01.org/mkl-dnn>`_ is already included in
                 the MXNet binary, you don't need to install it.

      .. admonition:: Command:

         .. container:: pip

            .. container:: native

               .. code-block:: bash

                  pip install mxnet

            .. container:: cuda

               .. code-block:: bash

                  # Here we assume CUDA 9.2 is installed. You can change the number
                  # according to your own CUDA version.
                  pip install mxnet-cu92

            .. container:: mkl-dnn

               .. code-block:: bash

                  pip install mxnet-mkl

            .. container:: cuda-mkl-dnn

               .. code-block:: bash

                  # Here we assume CUDA 9.2 is installed. You can change the number
                  # according to your own CUDA version.
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

.. raw:: html

   <style>.disabled { display: none; }</style>
   <script type="text/javascript" src='_static/install-options.js'></script>
