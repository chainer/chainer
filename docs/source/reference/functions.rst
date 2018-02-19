Standard Function implementations
=================================

.. module:: chainer.functions

Chainer provides basic :class:`~chainer.FunctionNode` implementations in the
:mod:`chainer.functions` package. Most of them are wrapped by plain Python
functions, which users should use.

.. note::
   As of v1.5, the concept of parameterized functions are gone, and they are
   replaced by corresponding :class:`~chainer.Link` implementations. They are
   found in the :mod:`~chainer.links` namespace.

..
   For contributors that want to update these lists:

   Each list corresponds to the package under chainer.functions. For example,
   the first section "Activation functions" shows functions under the
   chainer.functions.activation subpackage.

   KEEP EACH LIST IN LEXICOGRAPHICAL ORDER.


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
   chainer.functions.relu
   chainer.functions.selu
   chainer.functions.sigmoid
   chainer.functions.slstm
   chainer.functions.softmax
   chainer.functions.softplus
   chainer.functions.tanh
   chainer.functions.tree_lstm

Array manipulations
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.broadcast
   chainer.functions.broadcast_to
   chainer.functions.cast
   chainer.functions.concat
   chainer.functions.copy
   chainer.functions.depth2space
   chainer.functions.dstack
   chainer.functions.expand_dims
   chainer.functions.flatten
   chainer.functions.flip
   chainer.functions.fliplr
   chainer.functions.flipud
   chainer.functions.get_item
   chainer.functions.hstack
   chainer.functions.im2col
   chainer.functions.pad
   chainer.functions.pad_sequence
   chainer.functions.permutate
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
   chainer.functions.convolution_2d
   chainer.functions.convolution_nd
   chainer.functions.deconvolution_2d
   chainer.functions.deconvolution_nd
   chainer.functions.depthwise_convolution_2d
   chainer.functions.dilated_convolution_2d
   chainer.functions.embed_id
   chainer.functions.linear
   chainer.functions.n_step_bigru
   chainer.functions.n_step_bilstm
   chainer.functions.n_step_birnn
   chainer.functions.n_step_gru
   chainer.functions.n_step_lstm
   chainer.functions.n_step_rnn


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
   chainer.functions.cumsum
   chainer.functions.det
   chainer.functions.batch_det
   chainer.functions.erf
   chainer.functions.erfc
   chainer.functions.exp
   chainer.functions.expm1
   chainer.functions.fix
   chainer.functions.fmod
   chainer.functions.floor
   chainer.functions.identity
   chainer.functions.inv
   chainer.functions.linear_interpolate
   chainer.functions.log
   chainer.functions.log10
   chainer.functions.log1p
   chainer.functions.log2
   chainer.functions.logsumexp
   chainer.functions.matmul
   chainer.functions.max
   chainer.functions.maximum
   chainer.functions.mean
   chainer.functions.min
   chainer.functions.minimum
   chainer.functions.prod
   chainer.functions.rsqrt
   chainer.functions.scale
   chainer.functions.sin
   chainer.functions.sinh
   chainer.functions.sign
   chainer.functions.sqrt
   chainer.functions.square
   chainer.functions.squared_difference
   chainer.functions.sum
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
   chainer.functions.layer_normalization
   chainer.functions.local_response_normalization
   chainer.functions.normalize


Spatial pooling
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.average_pooling_2d
   chainer.functions.average_pooling_nd
   chainer.functions.max_pooling_2d
   chainer.functions.max_pooling_nd
   chainer.functions.roi_pooling_2d
   chainer.functions.spatial_pyramid_pooling_2d
   chainer.functions.unpooling_2d
   chainer.functions.unpooling_nd
   chainer.functions.upsampling_2d


Utility functions
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.functions.forget
