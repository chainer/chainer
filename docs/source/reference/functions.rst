.. _functions:

Standard Function implementations
=================================

.. module:: chainer.functions

Chainer provides basic :class:`~chainer.Function` implementations in the
:mod:`chainer.functions` module. All the functions are provided as plain
Python function that takes and returns:class:`Variable` objects (some also take
additional values as arguments).

.. note::
   Since v1.4, the concept of "parameterized functions" is gone and they are
   replaced by corresponding *link* implementations in the :mod:`chainer.links`
   module. They are still imported to the :mod:`chainer.functions` module with
   same names for compatibility. See :ref:`links` for the full list of provided
   links.


Array commputation functions
----------------------------
.. autofunction:: batch_matmul
.. autofunction:: bilinear
.. autofunction:: convolution_2d
.. autofunction:: embed_id
.. autofunction:: linear
.. autofunction:: matmul

Array manipulation functions
----------------------------
.. autofunction:: concat
.. autofunction:: copy
.. autofunction:: identity
.. autofunction:: parameter
.. autofunction:: reshape
.. autofunction:: split_axis

Activation functions
--------------------
.. autofunction:: clipped_relu
.. autofunction:: cos
.. autofunction:: exp
.. autofunction:: leaky_relu
.. autofunction:: log
.. autofunction:: lstm
.. autofunction:: prelu
.. autofunction:: relu
.. autofunction:: sigmoid
.. autofunction:: sin
.. autofunction:: softmax
.. autofunction:: softplus
.. autofunction:: tanh

Pooling functions
-----------------
.. autofunction:: average_pooling_2d
.. autofunction:: max_pooling_2d
.. autofunction:: spatial_pyramid_pooling_2d

Normalization functions
-----------------------
.. autofunction:: batch_normalization
.. autofunction:: local_response_normalization

Noise injecting functions 
-------------------------
.. autofunction:: dropout
.. autofunction:: gaussian

Loss, evaluation and aggregation
--------------------------------
.. autofunction:: accuracy
.. autofunction:: cross_covariance
.. autofunction:: mean_squared_error
.. autofunction:: negative_sampling
.. autofunction:: sigmoid_cross_entropy
.. autofunction:: softmax_cross_entropy
.. autofunction:: sum

Variational Auto-Encoder (VAE)
------------------------------
.. autofunction:: gaussian_kl_divergence
.. autofunction:: bernoulli_nll
.. autofunction:: gaussian_nll
