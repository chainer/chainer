Standard Function implementations
=================================

.. module:: chainer.functions

Chainer provides basic :class:`~chainer.Function` implementations in the
:mod:`chainer.functions` package. Most of them are wrapped by plain Python
functions, which users should use.

.. note::
   As of v1.5, the concept of parameterized functions are gone, and they are
   replaced by corresponding :class:`~chainer.Link` implementations. They are
   still put in the :mod:`~chainer.functions` namespace for backward
   compatibility, though it is strongly recommended to use them via the
   :mod:`chainer.links` package.

..
   For contributors that want to update these lists:

   Each list corresponds to the package under chainer.functions. For example,
   the first section "Activation functions" shows functions under the
   chainer.functions.activation subpackage.

   KEEP EACH LIST IN LEXICOGRAPHICAL ORDER.


Activation functions
--------------------

clipped_relu
~~~~~~~~~~~~
.. autofunction:: clipped_relu

crelu
~~~~~
.. autofunction:: crelu

elu
~~~
.. autofunction:: elu

hard_sigmoid
~~~~~~~~~~~~
.. autofunction:: hard_sigmoid

leaky_relu
~~~~~~~~~~
.. autofunction:: leaky_relu

log_softmax
~~~~~~~~~~~
.. autofunction:: log_softmax

lstm
~~~~
.. autofunction:: lstm

maxout
~~~~~~
.. autofunction:: maxout

prelu
~~~~~
.. autofunction:: prelu

relu
~~~~
.. autofunction:: relu

sigmoid
~~~~~~~
.. autofunction:: sigmoid

slstm
~~~~~
.. autofunction:: slstm

softmax
~~~~~~~
.. autofunction:: softmax

softplus
~~~~~~~~
.. autofunction:: softplus

tanh
~~~~
.. autofunction:: tanh


Array manipulations
-------------------

broadcast
~~~~~~~~~
.. autofunction:: broadcast

broadcast_to
~~~~~~~~~~~~
.. autofunction:: broadcast_to

cast
~~~~
.. autofunction:: cast

concat
~~~~~~
.. autofunction:: concat

copy
~~~~
.. autofunction:: copy

expand_dims
~~~~~~~~~~~
.. autofunction:: expand_dims

get_item
~~~~~~~~
.. autofunction:: get_item

hstack
~~~~~~~~
.. autofunction:: hstack

permutate
~~~~~~~~~
.. autofunction:: permutate

reshape
~~~~~~~
.. autofunction:: reshape

rollaxis
~~~~~~~~
.. autofunction:: rollaxis

select_item
~~~~~~~~~~~
.. autofunction:: select_item

separate
~~~~~~~~
.. autofunction:: separate

split_axis
~~~~~~~~~~
.. autofunction:: split_axis

stack
~~~~~
.. autofunction:: stack

swapaxes
~~~~~~~~
.. autofunction:: swapaxes

transpose
~~~~~~~~~
.. autofunction:: transpose

transpose_sequence
~~~~~~~~~~~~~~~~~~
.. autofunction:: transpose_sequence

where
~~~~~
.. autofunction:: where


Neural network connections
--------------------------

bilinear
~~~~~~~~
.. autofunction:: bilinear

convolution_2d
~~~~~~~~~~~~~~
.. autofunction:: convolution_2d

deconvolution_2d
~~~~~~~~~~~~~~~~
.. autofunction:: deconvolution_2d

embed_id
~~~~~~~~
.. autofunction:: embed_id

linear
~~~~~~
.. autofunction:: linear


Evaluation functions
--------------------

accuracy
~~~~~~~~
.. autofunction:: accuracy


Loss functions
--------------

bernoulli_nll
~~~~~~~~~~~~~
.. autofunction:: bernoulli_nll

connectionist_temporal_classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: connectionist_temporal_classification

contrastive
~~~~~~~~~~~
.. autofunction:: contrastive

crf1d
~~~~~
.. autofunction:: crf1d

cross_covariance
~~~~~~~~~~~~~~~~
.. autofunction:: cross_covariance

gaussian_kl_divergence
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: gaussian_kl_divergence

gaussian_nll
~~~~~~~~~~~~
.. autofunction:: gaussian_nll

hinge
~~~~~
.. autofunction:: hinge

huber_loss
~~~~~~~~~~
.. autofunction:: huber_loss

mean_squared_error
~~~~~~~~~~~~~~~~~~
.. autofunction:: mean_squared_error

negative_sampling
~~~~~~~~~~~~~~~~~
.. autofunction:: negative_sampling

sigmoid_cross_entropy
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: sigmoid_cross_entropy

softmax_cross_entropy
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: softmax_cross_entropy

triplet
~~~~~~~
.. autofunction:: triplet


Mathematical functions
----------------------

argmax
~~~~~~
.. autofunction:: argmax

argmin
~~~~~~
.. autofunction:: argmin

batch_inv
~~~~~~~~~
.. autofunction:: batch_inv

batch_l2_norm_squared
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: batch_l2_norm_squared

batch_matmul
~~~~~~~~~~~~
.. autofunction:: batch_matmul

bias
~~~~
.. autofunction:: bias

clip
~~~~
.. autofunction:: clip

cos
~~~
.. autofunction:: cos

exp
~~~
.. autofunction:: exp

identity
~~~~~~~~
.. autofunction:: identity

inv
~~~
.. autofunction:: inv

linear_interpolate
~~~~~~~~~~~~~~~~~~
.. autofunction:: linear_interpolate

log
~~~
.. autofunction:: log

log10
~~~~~
.. autofunction:: log10

log1p
~~~~~
.. autofunction:: log1p

log2
~~~~
.. autofunction:: log2

logsumexp
~~~~~~~~~
.. autofunction:: logsumexp

matmul
~~~~~~
.. autofunction:: matmul

max
~~~
.. autofunction:: max

maximum
~~~~~~~
.. autofunction:: maximum

min
~~~
.. autofunction:: min

minimum
~~~~~~~
.. autofunction:: minimum

rsqrt
~~~~~
.. autofunction:: rsqrt

scale
~~~~~
.. autofunction:: scale

sin
~~~
.. autofunction:: sin

sqrt
~~~~
.. autofunction:: sqrt

sum
~~~
.. autofunction:: sum

tan
~~~
.. autofunction:: tan


Noise injections
----------------

dropout
~~~~~~~
.. autofunction:: dropout

gaussian
~~~~~~~~
.. autofunction:: gaussian


Normalization functions
-----------------------

batch_normalization
~~~~~~~~~~~~~~~~~~~
.. autofunction:: batch_normalization

fixed_batch_normalization
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: fixed_batch_normalization

local_response_normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: local_response_normalization

normalize
~~~~~~~~~
.. autofunction:: normalize


Spatial pooling
---------------

average_pooling_2d
~~~~~~~~~~~~~~~~~~
.. autofunction:: average_pooling_2d

max_pooling_2d
~~~~~~~~~~~~~~
.. autofunction:: max_pooling_2d

roi_pooling_2d
~~~~~~~~~~~~~~
.. autofunction:: roi_pooling_2d

spatial_pyramid_pooling_2d
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: spatial_pyramid_pooling_2d

unpooling_2d
~~~~~~~~~~~~
.. autofunction:: unpooling_2d
