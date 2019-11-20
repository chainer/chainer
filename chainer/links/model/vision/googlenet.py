import collections
import os
import sys

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

import chainer
from chainer.dataset.convert import concat_examples
from chainer.dataset import download
from chainer import function
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.average import average
from chainer.functions.noise.dropout import dropout
from chainer.functions.normalization.local_response_normalization import (
    local_response_normalization)
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.functions.pooling.max_pooling_nd import max_pooling_2d
from chainer.initializers import constant
from chainer.initializers import uniform
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.inception import Inception
from chainer.links.connection.linear import Linear
from chainer.serializers import npz
from chainer.utils import argument
from chainer.utils import imgproc
from chainer.variable import Variable


class GoogLeNet(link.Chain):

    """A pre-trained GoogLeNet model provided by BVLC.

    When you specify the path of the pre-trained chainer model serialized as
    a ``.npz`` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    This model would be useful when you want to extract a semantic feature
    vector per image, or fine-tune the model on a different dataset.

    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.

    GoogLeNet, which is also called Inception-v1, is an architecture of
    convolutional neural network proposed in 2014. This model is relatively
    lightweight and requires small memory footprint during training compared
    with modern architectures such as ResNet. Therefore, if you fine-tune your
    network based on a model pre-trained by Imagenet and need to train it with
    large batch size, GoogLeNet may be useful. On the other hand, if you just
    want an off-the-shelf classifier, we recommend that you use ResNet50 or
    other models since they are more accurate than GoogLeNet.

    The original model is provided here:
    `<https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet>`_

    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically downloads the caffemodel from the internet.
            Note that in this case the converted chainer model is stored
            on ``$CHAINER_DATASET_ROOT/pfnet/chainer/models`` directory,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as a environment variable. The converted chainer model is
            automatically used from the second time.
            If the argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in BVLC, i.e.,
            ``chainer.initializers.LeCunUniform(scale=1.0)``.
            Note that, in Caffe, when weight_filler is specified as
            "xavier" type without variance_norm parameter, the weights are
            initialized by Uniform(-s, s), where
            :math:`s = \\sqrt{\\frac{3}{fan_{in}}}` and :math:`fan_{in}` is the
            number of input units. This corresponds to LeCunUniform in Chainer
            but not GlorotUniform.

    Attributes:
        available_layers (list of str): The list of available layer names
            used by ``forward`` and ``extract`` methods.

    """

    def __init__(self, pretrained_model='auto'):
        super(GoogLeNet, self).__init__()

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': constant.Zero()}
        else:
            # employ default initializers used in BVLC. For more detail, see
            # https://github.com/chainer/chainer/pull/2424#discussion_r109642209
            kwargs = {'initialW': uniform.LeCunUniform(scale=1.0)}

        with self.init_scope():
            self.conv1 = Convolution2D(3, 64, 7, stride=2, pad=3, **kwargs)
            self.conv2_reduce = Convolution2D(64, 64, 1, **kwargs)
            self.conv2 = Convolution2D(64, 192, 3, stride=1, pad=1, **kwargs)
            self.inc3a = Inception(192, 64, 96, 128, 16, 32, 32)
            self.inc3b = Inception(256, 128, 128, 192, 32, 96, 64)
            self.inc4a = Inception(480, 192, 96, 208, 16, 48, 64)
            self.inc4b = Inception(512, 160, 112, 224, 24, 64, 64)
            self.inc4c = Inception(512, 128, 128, 256, 24, 64, 64)
            self.inc4d = Inception(512, 112, 144, 288, 32, 64, 64)
            self.inc4e = Inception(528, 256, 160, 320, 32, 128, 128)
            self.inc5a = Inception(832, 256, 160, 320, 32, 128, 128)
            self.inc5b = Inception(832, 384, 192, 384, 48, 128, 128)
            self.loss3_fc = Linear(1024, 1000, **kwargs)

            self.loss1_conv = Convolution2D(512, 128, 1, **kwargs)
            self.loss1_fc1 = Linear(2048, 1024, **kwargs)
            self.loss1_fc2 = Linear(1024, 1000, **kwargs)

            self.loss2_conv = Convolution2D(528, 128, 1, **kwargs)
            self.loss2_fc1 = Linear(2048, 1024, **kwargs)
            self.loss2_fc2 = Linear(1024, 1000, **kwargs)

        if pretrained_model == 'auto':
            _retrieve(
                'bvlc_googlenet.npz',
                'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
                self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, relu]),
            ('pool1', [_max_pooling_2d, _local_response_normalization]),
            ('conv2_reduce', [self.conv2_reduce, relu]),
            ('conv2', [self.conv2, relu, _local_response_normalization]),
            ('pool2', [_max_pooling_2d]),
            ('inception_3a', [self.inc3a]),
            ('inception_3b', [self.inc3b]),
            ('pool3', [_max_pooling_2d]),
            ('inception_4a', [self.inc4a]),
            ('inception_4b', [self.inc4b]),
            ('inception_4c', [self.inc4c]),
            ('inception_4d', [self.inc4d]),
            ('inception_4e', [self.inc4e]),
            ('pool4', [_max_pooling_2d]),
            ('inception_5a', [self.inc5a]),
            ('inception_5b', [self.inc5b]),
            ('pool5', [_average_pooling_2d_k7]),
            ('loss3_fc', [_dropout, self.loss3_fc]),
            ('prob', [softmax]),
            # Since usually the following outputs are not used, they are put
            # after 'prob' to be skipped for efficiency.
            ('loss1_fc2', [_average_pooling_2d_k5, self.loss1_conv, relu,
                           self.loss1_fc1, relu, self.loss1_fc2]),
            ('loss2_fc2', [_average_pooling_2d_k5, self.loss2_conv, relu,
                           self.loss2_fc1, relu, self.loss2_fc2])
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.

        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None)
        _transfer_googlenet(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def forward(self, x, layers=None, **kwargs):
        """forward(self, x, layers=['prob'])

        Computes all the feature maps specified by ``layers``.

        Args:
            x (~chainer.Variable): Input variable. It should be prepared by
                ``prepare`` function.
            layers (list of str): The list of layer names you want to extract.

        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.

        """

        if layers is None:
            layers = ['prob']

        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs, train='train argument is not supported anymore. '
                'Use chainer.using_config')
            argument.assert_kwargs_empty(kwargs)

        h = x
        activations = {}
        inception_4a_cache = None
        inception_4d_cache = None
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if not target_layers:
                break

            if key == 'loss1_fc2':
                h = inception_4a_cache
            elif key == 'loss2_fc2':
                h = inception_4d_cache

            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)

            if key == 'inception_4a':
                inception_4a_cache = h
            elif key == 'inception_4d':
                inception_4d_cache = h

        return activations

    def extract(self, images, layers=None, size=(224, 224), **kwargs):
        """extract(self, images, layers=['pool5'], size=(224, 224))

        Extracts all the feature maps of given images.

        The difference of directly executing ``forward`` is that
        it directly accepts images as an input and automatically
        transforms them to a proper variable. That is,
        it is also interpreted as a shortcut method that implicitly calls
        ``prepare`` and ``forward`` functions.

        Unlike ``predict`` method, this method does not override
        ``chainer.config.train`` and ``chainer.config.enable_backprop``
        configuration. If you want to extract features without updating
        model parameters, you need to manually set configuration when
        calling this method as follows:

         .. code-block:: python

             # model is an instance of `GoogLeNet`
             with chainer.using_config('train', False):
                 with chainer.using_config('enable_backprop', False):
                     feature = model.extract([image])

        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
            layers (list of str): The list of layer names you want to extract.
            size (pair of ints): The resolution of resized images used as
                an input of CNN. All the given images are not resized
                if this argument is ``None``, but the resolutions of
                all the images should be the same.

        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.

        """

        if layers is None:
            layers = ['pool5']

        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs, train='train argument is not supported anymore. '
                'Use chainer.using_config',
                volatile='volatile argument is not supported anymore. '
                'Use chainer.using_config')
            argument.assert_kwargs_empty(kwargs)

        x = concat_examples([prepare(img, size=size) for img in images])
        x = Variable(self.xp.asarray(x))
        return self(x, layers=layers)

    def predict(self, images, oversample=True):
        """Computes all the probabilities of given images.

        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
                When you specify a color image as a :class:`numpy.ndarray`,
                make sure that color order is RGB.
            oversample (bool): If ``True``, it averages results across
                center, corners, and mirrors. Otherwise, it uses only the
                center.

        Returns:
            ~chainer.Variable: Output that contains the class probabilities
            of given images.

        """

        x = concat_examples([prepare(img, size=(256, 256)) for img in images])
        if oversample:
            x = imgproc.oversample(x, crop_dims=(224, 224))
        else:
            x = x[:, :, 16:240, 16:240]
        # Use no_backprop_mode to reduce memory consumption
        with function.no_backprop_mode(), chainer.using_config('train', False):
            x = Variable(self.xp.asarray(x))
            y = self(x, layers=['prob'])['prob']
            if oversample:
                n = len(y) // 10
                y_shape = y.shape[1:]
                y = reshape(y, (n, 10) + y_shape)
                y = average(y, axis=1)
        return y


def prepare(image, size=(224, 224)):
    """Converts the given image to the numpy array for GoogLeNet.

    Note that you have to call this method before ``forward``
    because the pre-trained GoogLeNet model requires to resize the given
    image, covert the RGB to the BGR, subtract the mean,
    and permute the dimensions before calling.

    Args:
        image (PIL.Image or numpy.ndarray): Input image.
            If an input is ``numpy.ndarray``, its shape must be
            ``(height, width)``, ``(height, width, channels)``,
            or ``(channels, height, width)``, and
            the order of the channels must be RGB.
        size (pair of ints): Size of converted images.
            If ``None``, the given image is not resized.

    Returns:
        numpy.ndarray: The converted output array.

    """

    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
    dtype = chainer.get_dtype()
    if isinstance(image, numpy.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0, :, :]
            elif image.shape[0] == 3:
                image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype(numpy.uint8))
    image = image.convert('RGB')
    if size:
        image = image.resize(size)
    image = numpy.asarray(image, dtype=dtype)
    image = image[:, :, ::-1]
    image -= numpy.array([104.0, 117.0, 123.0], dtype=dtype)  # BGR
    image = image.transpose((2, 0, 1))
    return image


def _transfer_inception(src, dst, names):
    for name in names:
        chain = getattr(dst, 'inc{}'.format(name))
        src_prefix = 'inception_{}/'.format(name)
        chain.conv1.W.array[:] = src[src_prefix + '1x1'].W.array
        chain.conv1.b.array[:] = src[src_prefix + '1x1'].b.array
        chain.proj3.W.array[:] = src[src_prefix + '3x3_reduce'].W.array
        chain.proj3.b.array[:] = src[src_prefix + '3x3_reduce'].b.array
        chain.conv3.W.array[:] = src[src_prefix + '3x3'].W.array
        chain.conv3.b.array[:] = src[src_prefix + '3x3'].b.array
        chain.proj5.W.array[:] = src[src_prefix + '5x5_reduce'].W.array
        chain.proj5.b.array[:] = src[src_prefix + '5x5_reduce'].b.array
        chain.conv5.W.array[:] = src[src_prefix + '5x5'].W.array
        chain.conv5.b.array[:] = src[src_prefix + '5x5'].b.array
        chain.projp.W.array[:] = src[src_prefix + 'pool_proj'].W.array
        chain.projp.b.array[:] = src[src_prefix + 'pool_proj'].b.array


def _transfer_googlenet(src, dst):
    # 1 #################################################################
    dst.conv1.W.array[:] = src['conv1/7x7_s2'].W.array
    dst.conv1.b.array[:] = src['conv1/7x7_s2'].b.array

    # 2 #################################################################
    dst.conv2_reduce.W.array[:] = src['conv2/3x3_reduce'].W.array
    dst.conv2_reduce.b.array[:] = src['conv2/3x3_reduce'].b.array
    dst.conv2.W.array[:] = src['conv2/3x3'].W.array
    dst.conv2.b.array[:] = src['conv2/3x3'].b.array

    # 3, 4, 5 ###########################################################
    _transfer_inception(src, dst, ['3a', '3b',
                                   '4a', '4b', '4c', '4d', '4e',
                                   '5a', '5b'])

    # outputs ############################################################
    dst.loss1_conv.W.array[:] = src['loss1/conv'].W.array
    dst.loss1_conv.b.array[:] = src['loss1/conv'].b.array
    dst.loss1_fc1.W.array[:] = src['loss1/fc'].W.array
    dst.loss1_fc1.b.array[:] = src['loss1/fc'].b.array
    dst.loss1_fc2.W.array[:] = src['loss1/classifier'].W.array
    dst.loss1_fc2.b.array[:] = src['loss1/classifier'].b.array

    dst.loss2_conv.W.array[:] = src['loss2/conv'].W.array
    dst.loss2_conv.b.array[:] = src['loss2/conv'].b.array
    dst.loss2_fc1.W.array[:] = src['loss2/fc'].W.array
    dst.loss2_fc1.b.array[:] = src['loss2/fc'].b.array
    dst.loss2_fc2.W.array[:] = src['loss2/classifier'].W.array
    dst.loss2_fc2.b.array[:] = src['loss2/classifier'].b.array

    dst.loss3_fc.W.array[:] = src['loss3/classifier'].W.array
    dst.loss3_fc.b.array[:] = src['loss3/classifier'].b.array


def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=3, stride=2)


def _local_response_normalization(x):
    return local_response_normalization(x, n=5, k=1, alpha=1e-4 / 5)


def _average_pooling_2d_k5(x):
    return average_pooling_2d(x, ksize=5, stride=3)


def _average_pooling_2d_k7(x):
    return average_pooling_2d(x, ksize=7, stride=1)


def _dropout(x):
    return dropout(x, ratio=0.4)


def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    sys.stderr.write(
        'Now loading caffemodel (usually it may take few minutes)\n')
    sys.stderr.flush()
    GoogLeNet.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name_npz, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))
