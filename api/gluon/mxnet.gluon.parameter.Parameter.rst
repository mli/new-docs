Parameter
=========

.. currentmodule:: mxnet.gluon.parameter

.. autoclass:: Parameter


   .. rubric:: Get and set parameters

   .. autosummary::
      :toctree: out_param

      Parameter.initialize
      Parameter.data
      Parameter.list_data
      Parameter.list_row_sparse_data
      Parameter.row_sparse_data
      Parameter.set_data
      Parameter.shape


   .. rubric:: Get and set gradients associated with parameters

   .. autosummary::
      :toctree: out_param

      Parameter.grad
      Parameter.list_grad
      Parameter.zero_grad
      Parameter.grad_req

   .. rubric:: Handle device contexts

   .. autosummary::
      :toctree: out_param

      Parameter.cast
      Parameter.list_ctx
      Parameter.reset_ctx

   .. rubric:: Convert to symbol

   .. autosummary::
      :toctree: out_param
      Parameter.var

   .. autosummary::
      :toctree: out_param

.. disqus::
