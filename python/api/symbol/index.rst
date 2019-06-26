mxnet.symbol
=========================

The Symbol API in Apache MXNet is an interface for symbolic programming. It features the use of computational graphs, reduced memory usage, and pre-use function optimization.


Example
-------

The following example shows how you might build a simple expression with the Symbol API.

.. code-block:: python

	import mxnet as mx
	# Two placeholers are created with mx.sym.variable
	a = mx.sym.Variable('a')
	b = mx.sym.Variable('b')
	# The symbol is constructed using the '+' operator
	c = a + b
	(a, b, c)


Tutorials
---------
.. container:: cards

   .. card::
      :title: Symbol API Intro
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/basic/symbol.html

      How to use MXNet's Symbol API.


Symbol Package
---------

.. container:: cards

   .. card::
      :title: Symbol
      :link: mxnet.symbol.Symbol.html

      Symbolic programming using the Symbol API.

.. toctree::
   :hidden:

   mxnet.symbol.Symbol

.. disqus::
