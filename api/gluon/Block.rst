Block
=====

.. currentmodule:: mxnet.gluon.nn

.. autoclass:: Block


   ..
      .. automethod:: __init__


   .. rubric:: Handle model parameters:

   .. autosummary::
      :toctree: out_container

      Block.initialize
      Block.save_parameters
      Block.load_parameters
      Block.collect_params
      Block.cast
      Block.apply

   .. rubric:: Run computation

   .. autosummary::
      :toctree: out_container

      Block.forward

   .. rubric:: Debugging

   .. autosummary::
      :toctree: out_container

      Block.summary

   .. rubric:: Advanced API for customization


   .. autosummary::
      :toctree: out_container

      Block.name_scope
      Block.register_child
      Block.register_forward_hook
      Block.register_forward_pre_hook

   .. rubric:: Attributes

   .. autosummary::

      Block.name
      Block.params
      Block.prefix


   .. warning::

      The following two APIs are deprecated since `v1.2.1
      <https://github.com/apache/incubator-mxnet/releases/tag/1.2.1>`_.

      .. autosummary::
          :toctree: out_container

          Block.save_params
          Block.load_params

.. disqus::
   :disqus_identifier: mxnet.gluon.nn.Gluon
