Link and Chains
===============

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
   chainer.links.Convolution1D
   chainer.links.Convolution2D
   chainer.links.Convolution3D
   chainer.links.ConvolutionND
   chainer.links.Deconvolution1D
   chainer.links.Deconvolution2D
   chainer.links.Deconvolution3D
   chainer.links.DeconvolutionND
   chainer.links.DeformableConvolution2D
   chainer.links.DepthwiseConvolution2D
   chainer.links.DilatedConvolution2D
   chainer.links.EmbedID
   chainer.links.GRU
   chainer.links.Highway
   chainer.links.Inception
   chainer.links.InceptionBN
   chainer.links.Linear
   chainer.links.LocalConvolution2D
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
   chainer.links.GroupNormalization
   chainer.links.LayerNormalization
   chainer.links.BinaryHierarchicalSoftmax
   chainer.links.BlackOut
   chainer.links.CRF1d
   chainer.links.SimplifiedDropconnect
   chainer.links.PReLU
   chainer.links.Swish
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

VGG Networks
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.VGG16Layers
   chainer.links.VGG19Layers
   chainer.links.model.vision.vgg.prepare

.. note::
   ChainerCV contains implementation of VGG networks as well (i.e.,
   :class:`chainercv.links.model.vgg.VGG16`). Unlike the Chainer's
   implementation, the ChainerCV's implementation
   assumes the color channel of the input image to be ordered in RGB instead
   of BGR.

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

.. note::
   ChainerCV contains implementation of ResNet as well (i.e.,
   :class:`chainercv.links.model.resnet.ResNet50`,
   :class:`chainercv.links.model.resnet.ResNet101`,
   :class:`chainercv.links.model.resnet.ResNet152`).
   Unlike the Chainer's
   implementation, the ChainerCV's implementation
   assumes the color channel of the input image to be ordered in RGB instead
   of BGR.


ChainerCV models
~~~~~~~~~~~~~~~~

.. note::
   ChainerCV supports implementations of links that are useful for computer
   vision problems, such as object detection, semantic segmentation, and
   instance segmentation.
   The documentation can be found in :mod:`chainercv.links`.
   Here is a subset of models with pre-trained weights supported by ChainerCV:

   * Detection
      * :class:`chainercv.links.model.faster_rcnn.FasterRCNNVGG16`
      * :class:`chainercv.links.model.ssd.SSD300`
      * :class:`chainercv.links.model.ssd.SSD512`
      * :class:`chainercv.links.model.yolo.YOLOv2`
      * :class:`chainercv.links.model.yolo.YOLOv3`
   * Semantic Segmentation
      * :class:`chainercv.links.model.segnet.SegNetBasic`
      * :class:`chainercv.experimental.links.model.pspnet.PSPNetResNet101`
   * Instance Segmentation
      * :class:`chainercv.experimental.links.model.fcis.FCISResNet101`
   * Classification
      * :class:`chainercv.links.model.resnet.ResNet101`
      * :class:`chainercv.links.model.resnet.ResNet152`
      * :class:`chainercv.links.model.resnet.ResNet50`
      * :class:`chainercv.links.model.senet.SEResNet101`
      * :class:`chainercv.links.model.senet.SEResNet152`
      * :class:`chainercv.links.model.senet.SEResNet50`
      * :class:`chainercv.links.model.senet.SEResNeXt101`
      * :class:`chainercv.links.model.senet.SEResNeXt50`
      * :class:`chainercv.links.model.vgg.VGG16`

Compatibility with other frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.TheanoFunction
   chainer.links.caffe.CaffeFunction

Link and Chain base classes
---------------------------

.. module:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Link
   chainer.Chain
   chainer.ChainList
   chainer.Sequential

Link hooks
--------------

.. module:: chainer.link_hooks

Chainer provides a link-hook mechanism that enriches the behavior of :class:`~chainer.Link`.

.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.link_hooks.SpectralNormalization
   chainer.link_hooks.TimerHook

You can also implement your own link-hook to inject arbitrary code before/after the forward propagation.

.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.LinkHook
