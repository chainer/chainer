Standard Link implementations
=============================

.. module:: chainer.links

Chainer provides many :class:`~chainer.Link` implementations in the
:mod:`chainer.links` package.

.. note::
   Some of the links are originally defined in the :mod:`chainer.functions`
   namespace. They are still left in the namespace for backward compatibility,
   though it is strongly recommended to use them via the :mod:`chainer.links`
   package.


Learnable connections
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.Bias
   chainer.links.Bilinear
   chainer.links.ChildSumTreeLSTM
   chainer.links.Convolution2D
   chainer.links.ConvolutionND
   chainer.links.Deconvolution2D
   chainer.links.DeconvolutionND
   chainer.links.DepthwiseConvolution2D
   chainer.links.DilatedConvolution2D
   chainer.links.EmbedID
   chainer.links.GRU
   chainer.links.Highway
   chainer.links.Inception
   chainer.links.InceptionBN
   chainer.links.Linear
   chainer.links.LSTM
   chainer.links.MLPConvolution2D
   chainer.links.NaryTreeLSTM
   chainer.links.NStepBiGRU
   chainer.links.NStepBiLSTM
   chainer.links.NStepBiRNNReLU
   chainer.links.NStepBiRNNTanh
   chainer.links.NStepGRU
   chainer.links.NStepLSTM
   chainer.links.NStepRNNReLU
   chainer.links.NStepRNNTanh
   chainer.links.Parameter
   chainer.links.Scale
   chainer.links.StatefulGRU
   chainer.links.StatelessGRU
   chainer.links.StatefulMGU
   chainer.links.StatelessMGU
   chainer.links.StatefulPeepholeLSTM
   chainer.links.StatefulZoneoutLSTM
   chainer.links.StatelessLSTM

Activation/loss/normalization functions with parameters
-------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.BatchNormalization
   chainer.links.BatchRenormalization
   chainer.links.LayerNormalization
   chainer.links.BinaryHierarchicalSoftmax
   chainer.links.BlackOut
   chainer.links.CRF1d
   chainer.links.SimplifiedDropconnect
   chainer.links.PReLU
   chainer.links.Maxout
   chainer.links.NegativeSampling

Machine learning models
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.Classifier

Pre-trained models
------------------

Pre-trained models are mainly used to achieve a good performance with a small
dataset, or extract a semantic feature vector. Although ``CaffeFunction``
automatically loads a pre-trained model released as a caffemodel,
the following link models provide an interface for automatically converting
caffemodels, and easily extracting semantic feature vectors.

For example, to extract the feature vectors with ``VGG16Layers``, which is
a common pre-trained model in the field of image recognition,
users need to write the following few lines::

    from chainer.links import VGG16Layers
    from PIL import Image

    model = VGG16Layers()
    img = Image.open("path/to/image.jpg")
    feature = model.extract([img], layers=["fc7"])["fc7"]

where ``fc7`` denotes a layer before the last fully-connected layer.
Unlike the usual links, these classes automatically load all the
parameters from the pre-trained models during initialization.

VGG16Layers
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.VGG16Layers
   chainer.links.model.vision.vgg.prepare

GoogLeNet
~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.GoogLeNet
   chainer.links.model.vision.googlenet.prepare

Residual Networks
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.model.vision.resnet.ResNetLayers
   chainer.links.ResNet50Layers
   chainer.links.ResNet101Layers
   chainer.links.ResNet152Layers
   chainer.links.model.vision.resnet.prepare

Compatibility with other frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.TheanoFunction
   chainer.links.caffe.CaffeFunction
