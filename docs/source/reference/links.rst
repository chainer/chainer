.. _links:

Standard Link Implementations
=============================

.. module:: chainer.links

Chainer provides basic :class:`~chainer.Link` implementations in the
:mod:`chainer.links` module.

.. note::
   The Link class is introduced at v1.4, where the "parameterized functions" are
   moved from the :mod:`chainer.functions` module to the :mod:`chainer.links`
   module. They are still left in the functions module, too.


Learnable connections
---------------------
.. autoclass:: Bilinear
   :members:
.. autoclass:: Convolution2D
   :members:
.. autoclass:: EmbedID
   :members:
.. autoclass:: Inception
   :members:
.. autoclass:: InceptionBN
   :members:
.. autoclass:: Linear
   :members:
.. autoclass:: LSTM
   :members:
.. autoclass:: MLPConvolution2D
   :members:
.. autoclass:: Parameter
   :members:

Learnable activation functions
------------------------------
.. autoclass:: PReLU
   :members:

Loss layers with structures
---------------------------
.. autoclass:: BinaryHierarchicalSoftmax
   :members:
.. autoclass:: NegativeSampling
   :members:

Learnable normalization
-----------------------
.. autoclass:: BatchNormalization
   :members:
