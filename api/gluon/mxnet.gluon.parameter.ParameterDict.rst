ParameterDict
=============

.. currentmodule:: mxnet.gluon.parameter

.. autoclass:: ParameterDict

   .. rubric:: Load and save parameters

   .. autosummary::
      :toctree: out_param

      ParameterDict.load
      ParameterDict.save

   .. rubric:: Get a particular parameter

   .. autosummary::
      :toctree: out_param

      ParameterDict.get
      ParameterDict.get_constant

   .. rubric:: Get (name, paramter) pairs

   .. autosummary::
      :toctree: out_param

      ParameterDict.items
      ParameterDict.keys
      ParameterDict.values

   .. rubric:: Update parameters

   .. autosummary::
      :toctree: out_param

      ParameterDict.initialize
      ParameterDict.setattr
      ParameterDict.update


   .. rubric:: Set devices contexts and gradients

   .. autosummary::
      :toctree: out_param

      ParameterDict.reset_ctx
      ParameterDict.zero_grad

   .. rubric:: Attributes

   .. autosummary::

      ParameterDict.prefix

.. disqus::
