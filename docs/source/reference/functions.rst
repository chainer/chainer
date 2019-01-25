Functions
=========

.. module:: chainer.functions

Chainer provides variety of built-in function implementations in :mod:`chainer.functions` package.
These functions usually return a :class:`~chainer.Variable` object or a tuple of multiple :class:`~chainer.Variable` objects.
For a :class:`~chainer.Variable` argument of a function, an :ref:`ndarray` can be passed if you do not need its gradient.
Some functions additionally supports scalar arguments.

.. note::
    Functions implemented in Chainer consists of the following two parts:

    * A class that inherits :class:`~chainer.FunctionNode`, which defines forward/backward computation.
    * A "wrapper" function around the class.

    APIs listed in this page are "wrapper" of :class:`~chainer.FunctionNode` implementations.
    In most cases, you don't have to use :class:`~chainer.FunctionNode` classes directly.

    For example, :func:`chainer.functions.sum` is a wrapper function defined as ``def sum(...):`` in `chainer/functions/math/sum.py <https://github.com/chainer/chainer/blob/master/chainer/functions/math/sum.py>`__, and it calls its corresponding :class:`~chainer.FunctionNode` implementation, ``Sum``.
    Some functions may not have the corresponding :class:`~chainer.FunctionNode` implementation; one example is :func:`chainer.functions.average`, which is defined in `chainer/functions/math/average.py <https://github.com/chainer/chainer/blob/master/chainer/functions/math/average.py>`__, which calls other wrapper functions to calculate average.

    If you are implementing your own functions, please see :doc:`../guides/functions`.

..
   For contributors that want to update these lists:

   Each list corresponds to the package under chainer.functions. For example,
   the first section "Activation functions" shows functions under the
   chainer.functions.activation subpackage.

   KEEP EACH LIST IN LEXICOGRAPHICAL ORDER.


Arithmetic functions
--------------------

Basic arithmetic operations for :class:`~chainer.Variable`\s are implemented as operators.
Refer to the Notes section of :class:`~chainer.Variable` for details.

:func:`chainer.functions.add` provides better performance when accumulating three or more :class:`~chainer.Variable`\s at once.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.add

Activation functions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.clipped_relu
   chainer.functions.crelu
   chainer.functions.elu
   chainer.functions.hard_sigmoid
   chainer.functions.leaky_relu
   chainer.functions.log_softmax
   chainer.functions.lstm
   chainer.functions.maxout
   chainer.functions.prelu
   chainer.functions.rrelu
   chainer.functions.relu
   chainer.functions.selu
   chainer.functions.sigmoid
   chainer.functions.slstm
   chainer.functions.softmax
   chainer.functions.softplus
   chainer.functions.swish
   chainer.functions.tanh
   chainer.functions.tree_lstm

Array manipulations
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.as_strided
   chainer.functions.broadcast
   chainer.functions.broadcast_to
   chainer.functions.cast
   chainer.functions.concat
   chainer.functions.copy
   chainer.functions.depth2space
   chainer.functions.diagonal
   chainer.functions.dstack
   chainer.functions.expand_dims
   chainer.functions.flatten
   chainer.functions.flip
   chainer.functions.fliplr
   chainer.functions.flipud
   chainer.functions.get_item
   chainer.functions.hstack
   chainer.functions.im2col
   chainer.functions.moveaxis
   chainer.functions.pad
   chainer.functions.pad_sequence
   chainer.functions.permutate
   chainer.functions.repeat
   chainer.functions.reshape
   chainer.functions.resize_images
   chainer.functions.rollaxis
   chainer.functions.scatter_add
   chainer.functions.select_item
   chainer.functions.separate
   chainer.functions.space2depth
   chainer.functions.spatial_transformer_grid
   chainer.functions.spatial_transformer_sampler
   chainer.functions.split_axis
   chainer.functions.squeeze
   chainer.functions.stack
   chainer.functions.swapaxes
   chainer.functions.tile
   chainer.functions.transpose
   chainer.functions.transpose_sequence
   chainer.functions.vstack
   chainer.functions.where

Neural network connections
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.bilinear
   chainer.functions.convolution_1d
   chainer.functions.convolution_2d
   chainer.functions.convolution_3d
   chainer.functions.convolution_nd
   chainer.functions.deconvolution_1d
   chainer.functions.deconvolution_2d
   chainer.functions.deconvolution_3d
   chainer.functions.deconvolution_nd
   chainer.functions.depthwise_convolution_2d
   chainer.functions.deformable_convolution_2d_sampler
   chainer.functions.dilated_convolution_2d
   chainer.functions.embed_id
   chainer.functions.linear
   chainer.functions.local_convolution_2d
   chainer.functions.n_step_bigru
   chainer.functions.n_step_bilstm
   chainer.functions.n_step_birnn
   chainer.functions.n_step_gru
   chainer.functions.n_step_lstm
   chainer.functions.n_step_rnn
   chainer.functions.shift


Evaluation functions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.accuracy
   chainer.functions.binary_accuracy
   chainer.functions.classification_summary
   chainer.functions.f1_score
   chainer.functions.precision
   chainer.functions.r2_score
   chainer.functions.recall


Loss functions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.absolute_error
   chainer.functions.bernoulli_nll
   chainer.functions.black_out
   chainer.functions.connectionist_temporal_classification
   chainer.functions.contrastive
   chainer.functions.crf1d
   chainer.functions.argmax_crf1d
   chainer.functions.cross_covariance
   chainer.functions.decov
   chainer.functions.discriminative_margin_based_clustering_loss
   chainer.functions.gaussian_kl_divergence
   chainer.functions.gaussian_nll
   chainer.functions.hinge
   chainer.functions.huber_loss
   chainer.functions.mean_absolute_error
   chainer.functions.mean_squared_error
   chainer.functions.negative_sampling
   chainer.functions.sigmoid_cross_entropy
   chainer.functions.softmax_cross_entropy
   chainer.functions.squared_error
   chainer.functions.triplet

Mathematical functions
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.absolute
   chainer.functions.arccos
   chainer.functions.arcsin
   chainer.functions.arctan
   chainer.functions.arctan2
   chainer.functions.argmax
   chainer.functions.argmin
   chainer.functions.average
   chainer.functions.batch_inv
   chainer.functions.batch_l2_norm_squared
   chainer.functions.batch_matmul
   chainer.functions.bias
   chainer.functions.ceil
   chainer.functions.clip
   chainer.functions.cos
   chainer.functions.cosh
   chainer.functions.cumprod
   chainer.functions.cumsum
   chainer.functions.det
   chainer.functions.batch_det
   chainer.functions.digamma
   chainer.functions.einsum
   chainer.functions.erf
   chainer.functions.erfc
   chainer.functions.erfcinv
   chainer.functions.erfcx
   chainer.functions.erfinv
   chainer.functions.exp
   chainer.functions.expm1
   chainer.functions.fft
   chainer.functions.fix
   chainer.functions.fmod
   chainer.functions.floor
   chainer.functions.identity
   chainer.functions.ifft
   chainer.functions.inv
   chainer.functions.lgamma
   chainer.functions.linear_interpolate
   chainer.functions.log
   chainer.functions.log10
   chainer.functions.log1p
   chainer.functions.log2
   chainer.functions.log_ndtr
   chainer.functions.logsumexp
   chainer.functions.matmul
   chainer.functions.max
   chainer.functions.maximum
   chainer.functions.mean
   chainer.functions.min
   chainer.functions.minimum
   chainer.functions.ndtr
   chainer.functions.ndtri
   chainer.functions.prod
   chainer.functions.polygamma
   chainer.functions.rsqrt
   chainer.functions.scale
   chainer.functions.sin
   chainer.functions.sinh
   chainer.functions.sign
   chainer.functions.sparse_matmul
   chainer.functions.sqrt
   chainer.functions.square
   chainer.functions.squared_difference
   chainer.functions.sum
   chainer.functions.sum_to
   chainer.functions.tanh
   chainer.functions.tan
   chainer.functions.tensordot

Noise injections
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.dropout
   chainer.functions.gaussian
   chainer.functions.gumbel_softmax
   chainer.functions.simplified_dropconnect
   chainer.functions.zoneout

Normalization functions
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.batch_normalization
   chainer.functions.batch_renormalization
   chainer.functions.fixed_batch_normalization
   chainer.functions.fixed_batch_renormalization
   chainer.functions.group_normalization
   chainer.functions.layer_normalization
   chainer.functions.local_response_normalization
   chainer.functions.normalize


Spatial pooling
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.average_pooling_1d
   chainer.functions.average_pooling_2d
   chainer.functions.average_pooling_3d
   chainer.functions.average_pooling_nd
   chainer.functions.max_pooling_1d
   chainer.functions.max_pooling_2d
   chainer.functions.max_pooling_3d
   chainer.functions.max_pooling_nd
   chainer.functions.roi_average_align_2d
   chainer.functions.roi_average_pooling_2d
   chainer.functions.roi_max_align_2d
   chainer.functions.roi_max_pooling_2d
   chainer.functions.roi_pooling_2d
   chainer.functions.spatial_pyramid_pooling_2d
   chainer.functions.unpooling_1d
   chainer.functions.unpooling_2d
   chainer.functions.unpooling_3d
   chainer.functions.unpooling_nd
   chainer.functions.upsampling_2d


Utility functions
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.forget

Function base
-------------

.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Function
   chainer.FunctionAdapter
   chainer.FunctionNode
   chainer.force_backprop_mode
   chainer.no_backprop_mode
   chainer.grad

Function hooks
--------------

.. module:: chainer.function_hooks

Chainer provides a function-hook mechanism that enriches the behavior of forward and backward propagation of :class:`~chainer.FunctionNode` and :class:`~chainer.Function`.

.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.function_hooks.CUDAProfileHook
   chainer.function_hooks.CupyMemoryProfileHook
   chainer.function_hooks.PrintHook
   chainer.function_hooks.TimerHook

You can also implement your own function-hook to inject arbitrary code before/after the forward/backward propagation.

.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.FunctionHook
