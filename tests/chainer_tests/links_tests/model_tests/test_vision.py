import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer.links.model.vision import googlenet
from chainer.links.model.vision import resnet
from chainer.links.model.vision import vgg
from chainer import testing
from chainer.testing import attr
from chainer.variable import Variable


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'n_layers': [50, 101, 152],
    'downsample_fb': [True, False],
}) + [{
    'dtype': numpy.float16,
    'n_layers': 50,
    'downsample_fb': False,
}])
@unittest.skipUnless(resnet.available, 'Pillow is required')
@attr.slow
class TestResNetLayers(unittest.TestCase):

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()

        if self.n_layers == 50:
            self.link = resnet.ResNet50Layers(
                pretrained_model=None, downsample_fb=self.downsample_fb)
        elif self.n_layers == 101:
            self.link = resnet.ResNet101Layers(
                pretrained_model=None, downsample_fb=self.downsample_fb)
        elif self.n_layers == 152:
            self.link = resnet.ResNet152Layers(
                pretrained_model=None, downsample_fb=self.downsample_fb)

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def test_available_layers(self):
        result = self.link.available_layers
        assert isinstance(result, list)
        assert len(result) == 9

    def check_call(self):
        xp = self.link.xp

        # Suppress warning that arises from zero division in BatchNormalization
        with numpy.errstate(divide='ignore'):
            x1 = Variable(xp.asarray(numpy.random.uniform(
                -1, 1, (1, 3, 224, 224)).astype(self.dtype)))
            y1 = cuda.to_cpu(self.link(x1)['prob'].data)
            assert y1.shape == (1, 1000)

            x2 = Variable(xp.asarray(numpy.random.uniform(
                -1, 1, (1, 3, 128, 128)).astype(self.dtype)))
            y2 = cuda.to_cpu(self.link(x2, layers=['pool5'])['pool5'].data)
            assert y2.shape == (1, 2048)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def test_prepare(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        x3 = numpy.random.uniform(0, 255, (160, 120, 3)).astype(self.dtype)
        x4 = numpy.random.uniform(0, 255, (1, 160, 120)).astype(self.dtype)
        x5 = numpy.random.uniform(0, 255, (3, 160, 120)).astype(numpy.uint8)

        y1 = resnet.prepare(x1)
        assert y1.shape == (3, 224, 224)
        assert y1.dtype == self.dtype
        y2 = resnet.prepare(x2)
        assert y2.shape == (3, 224, 224)
        assert y2.dtype == self.dtype
        y3 = resnet.prepare(x3, size=None)
        assert y3.shape == (3, 160, 120)
        assert y3.dtype == self.dtype
        y4 = resnet.prepare(x4)
        assert y4.shape == (3, 224, 224)
        assert y4.dtype == self.dtype
        y5 = resnet.prepare(x5, size=None)
        assert y5.shape == (3, 160, 120)
        assert y5.dtype == self.dtype

    def check_extract(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)

        with numpy.errstate(divide='ignore'):
            result = self.link.extract([x1, x2], layers=['res3', 'pool5'])
            assert len(result) == 2
            y1 = cuda.to_cpu(result['res3'].data)
            assert y1.shape == (2, 512, 28, 28)
            assert y1.dtype == self.dtype
            y2 = cuda.to_cpu(result['pool5'].data)
            assert y2.shape == (2, 2048)
            assert y2.dtype == self.dtype

            x3 = numpy.random.uniform(0, 255, (80, 60)).astype(numpy.uint8)
            result = self.link.extract([x3], layers=['res2'], size=None)
            assert len(result) == 1
            y3 = cuda.to_cpu(result['res2'].data)
            assert y3.shape == (1, 256, 20, 15)
            assert y3.dtype == self.dtype

    def test_extract_cpu(self):
        err = 'ignore' if self.dtype is numpy.float16 else None
        with numpy.errstate(over=err):  # ignore FP16 overflow
            self.check_extract()

    @attr.gpu
    def test_extract_gpu(self):
        self.link.to_gpu()
        self.check_extract()

    def check_predict(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)

        with numpy.errstate(divide='ignore'):
            result = self.link.predict([x1, x2], oversample=False)
            y = cuda.to_cpu(result.data)
            assert y.shape == (2, 1000)
            assert y.dtype == self.dtype
            result = self.link.predict([x1, x2], oversample=True)
            y = cuda.to_cpu(result.data)
            assert y.shape == (2, 1000)
            assert y.dtype == self.dtype

    def test_predict_cpu(self):
        err = 'ignore' if self.dtype is numpy.float16 else None
        with numpy.errstate(over=err):  # ignore FP16 overflow
            self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()

    def check_copy(self):
        copied = self.link.copy()

        assert copied.conv1 is copied.functions['conv1'][0]
        assert (
            copied.res2.a is
            getattr(copied.res2, copied.res2._forward[0]))

    def test_copy_cpu(self):
        self.check_copy()

    @attr.gpu
    def test_copy_gpu(self):
        self.link.to_gpu()
        self.check_copy()


@testing.parameterize(*testing.product({
    'n_layers': [16, 19],
    'dtype': [numpy.float16, numpy.float32],
}))
@unittest.skipUnless(resnet.available, 'Pillow is required')
@attr.slow
class TestVGGs(unittest.TestCase):

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()
        if self.n_layers == 16:
            self.link = vgg.VGG16Layers(pretrained_model=None)
        elif self.n_layers == 19:
            self.link = vgg.VGG19Layers(pretrained_model=None)

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def test_available_layers(self):
        result = self.link.available_layers
        assert isinstance(result, list)
        if self.n_layers == 16:
            assert len(result) == 22
        elif self.n_layers == 19:
            assert len(result) == 25

    def check_call(self):
        xp = self.link.xp

        x1 = Variable(xp.asarray(numpy.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(self.dtype)))
        y1 = cuda.to_cpu(self.link(x1)['prob'].data)
        assert y1.shape == (1, 1000)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def test_prepare(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        x3 = numpy.random.uniform(0, 255, (160, 120, 3)).astype(self.dtype)
        x4 = numpy.random.uniform(0, 255, (1, 160, 120)).astype(self.dtype)
        x5 = numpy.random.uniform(0, 255, (3, 160, 120)).astype(numpy.uint8)

        y1 = vgg.prepare(x1)
        assert y1.shape == (3, 224, 224)
        assert y1.dtype == self.dtype
        y2 = vgg.prepare(x2)
        assert y2.shape == (3, 224, 224)
        assert y2.dtype == self.dtype
        y3 = vgg.prepare(x3, size=None)
        assert y3.shape == (3, 160, 120)
        assert y3.dtype == self.dtype
        y4 = vgg.prepare(x4)
        assert y4.shape == (3, 224, 224)
        assert y4.dtype == self.dtype
        y5 = vgg.prepare(x5, size=None)
        assert y5.shape == (3, 160, 120)
        assert y5.dtype == self.dtype

    def check_extract(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        result = self.link.extract([x1, x2], layers=['pool3', 'fc7'])
        assert len(result) == 2
        y1 = cuda.to_cpu(result['pool3'].data)
        assert y1.shape == (2, 256, 28, 28)
        assert y1.dtype == self.dtype
        y2 = cuda.to_cpu(result['fc7'].data)
        assert y2.shape == (2, 4096)
        assert y2.dtype == self.dtype

        x3 = numpy.random.uniform(0, 255, (80, 60)).astype(numpy.uint8)
        result = self.link.extract([x3], layers=['pool1'], size=None)
        assert len(result) == 1
        y3 = cuda.to_cpu(result['pool1'].data)
        assert y3.shape == (1, 64, 40, 30)
        assert y3.dtype == self.dtype

    def test_extract_cpu(self):
        self.check_extract()

    @attr.gpu
    def test_extract_gpu(self):
        self.link.to_gpu()
        self.check_extract()

    def check_predict(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        result = self.link.predict([x1, x2], oversample=False)
        y = cuda.to_cpu(result.data)
        assert y.shape == (2, 1000)
        assert y.dtype == self.dtype
        result = self.link.predict([x1, x2], oversample=True)
        y = cuda.to_cpu(result.data)
        assert y.shape == (2, 1000)
        assert y.dtype == self.dtype

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()

    def check_copy(self):
        copied = self.link.copy()

        assert copied.conv1_1 is copied.functions['conv1_1'][0]

    def test_copy_cpu(self):
        self.check_copy()

    @attr.gpu
    def test_copy_gpu(self):
        self.link.to_gpu()
        self.check_copy()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32],
}))
@unittest.skipUnless(googlenet.available, 'Pillow is required')
@attr.slow
class TestGoogLeNet(unittest.TestCase):

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()
        self.link = googlenet.GoogLeNet(pretrained_model=None)

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def test_available_layers(self):
        result = self.link.available_layers
        assert isinstance(result, list)
        assert len(result) == 21

    def check_call_prob(self):
        xp = self.link.xp

        x = Variable(xp.asarray(numpy.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(self.dtype)))
        y = cuda.to_cpu(self.link(x)['prob'].data)
        assert y.shape == (1, 1000)

    def check_call_loss1_fc2(self):
        xp = self.link.xp

        x = Variable(xp.asarray(numpy.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(self.dtype)))
        y = cuda.to_cpu(self.link(x, ['loss1_fc2'])['loss1_fc2'].data)
        assert y.shape == (1, 1000)

    def check_call_loss2_fc2(self):
        xp = self.link.xp

        x = Variable(xp.asarray(numpy.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(self.dtype)))
        y = cuda.to_cpu(self.link(x, ['loss2_fc2'])['loss2_fc2'].data)
        assert y.shape == (1, 1000)

    def test_call_cpu(self):
        self.check_call_prob()
        self.check_call_loss1_fc2()
        self.check_call_loss2_fc2()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call_prob()
        self.check_call_loss1_fc2()
        self.check_call_loss2_fc2()

    def test_prepare(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        x3 = numpy.random.uniform(0, 255, (160, 120, 3)).astype(self.dtype)
        x4 = numpy.random.uniform(0, 255, (1, 160, 120)).astype(self.dtype)
        x5 = numpy.random.uniform(0, 255, (3, 160, 120)).astype(numpy.uint8)

        y1 = googlenet.prepare(x1)
        assert y1.shape == (3, 224, 224)
        assert y1.dtype, self.dtype
        y2 = googlenet.prepare(x2)
        assert y2.shape == (3, 224, 224)
        assert y2.dtype, self.dtype
        y3 = googlenet.prepare(x3, size=None)
        assert y3.shape == (3, 160, 120)
        assert y3.dtype, self.dtype
        y4 = googlenet.prepare(x4)
        assert y4.shape == (3, 224, 224)
        assert y4.dtype, self.dtype
        y5 = googlenet.prepare(x5, size=None)
        assert y5.shape == (3, 160, 120)
        assert y5.dtype, self.dtype

    def check_extract(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)

        result = self.link.extract([x1, x2], layers=['pool5', 'loss3_fc'])
        assert len(result) == 2
        y1 = cuda.to_cpu(result['pool5'].data)
        assert y1.shape == (2, 1024, 1, 1)
        assert y1.dtype == self.dtype
        y2 = cuda.to_cpu(result['loss3_fc'].data)
        assert y2.shape == (2, 1000)
        assert y2.dtype == self.dtype

        x3 = numpy.random.uniform(0, 255, (80, 60)).astype(numpy.uint8)
        result = self.link.extract([x3], layers=['pool1'], size=None)
        assert len(result) == 1
        y3 = cuda.to_cpu(result['pool1'].data)
        assert y3.shape == (1, 64, 20, 15)
        assert y3.dtype == self.dtype

    def test_extract_cpu(self):
        err = 'ignore' if self.dtype is numpy.float16 else None
        with numpy.errstate(over=err):  # ignore FP16 overflow
            self.check_extract()

    @attr.gpu
    def test_extract_gpu(self):
        self.link.to_gpu()
        self.check_extract()

    def check_predict(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)

        result = self.link.predict([x1, x2], oversample=False)
        y = cuda.to_cpu(result.data)
        assert y.shape == (2, 1000)
        assert y.dtype == self.dtype
        result = self.link.predict([x1, x2], oversample=True)
        y = cuda.to_cpu(result.data)
        assert y.shape == (2, 1000)
        assert y.dtype == self.dtype

    def test_predict_cpu(self):
        err = 'ignore' if self.dtype is numpy.float16 else None
        with numpy.errstate(over=err):  # ignore FP16 overflow
            self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()

    def check_copy(self):
        copied = self.link.copy()

        assert copied.conv1 is copied.functions['conv1'][0]

    def test_copy_cpu(self):
        self.check_copy()

    @attr.gpu
    def test_copy_gpu(self):
        self.link.to_gpu()
        self.check_copy()


testing.run_module(__name__, __file__)
