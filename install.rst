Install
=======

.. raw:: html

    <script type="text/javascript" src='../_static/install-options.js'></script>
    <script type="text/javascript" src='_static/install-options.js'></script>

.. container:: install

   .. container:: opt-group

      .. container:: title

         OS:

      .. container:: option

         Linux

      .. container:: option

         Macos

      .. container:: option

         Windows

   .. container:: opt-group

      .. container:: title

         Package:

      .. container:: option active

         pip

      .. container:: option

         docker

      .. container:: option

         source


   .. container:: opt-group

      .. container:: title

         Backend:

      .. container:: option active

         Native

      .. container:: option

         CUDA 8.0

      .. container:: option

         CUDA 9.0

      .. container:: option

         CUDA 9.2

      .. container:: option

         MKL-DNN


   .. container:: pip

      .. container:: linux windows macos

         .. container:: native

            .. code-block:: bash

               pip install mxnet
               # you can append --pre tag to install the nightly version.

         .. container:: mkl-dnn

            .. code-block:: bash

               pip install mxnet-mkldnn

      .. container:: linux windows

         .. container:: cuda-8-0

            .. code-block:: bash

               pip install mxnet-cu80

         .. container:: cuda-9-0

            .. code-block:: bash

               pip install mxnet-cu90

         .. container:: cuda-9-2

            .. code-block:: bash

               pip install mxnet-cu90

      .. container:: macos

         .. container:: cuda-8-0 cuda-9-0 cuda-9-2

            .. code-block:: bash

               # CUDA version is not supported in Macos
