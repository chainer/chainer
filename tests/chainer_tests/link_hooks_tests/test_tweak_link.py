import unittest

import numpy
import pytest

import chainer
import chainer.links as L
from chainer import link_hooks
from chainer import testing


NAME = 'NAME'
SHAPE = (100, 100)
DTYPE = numpy.float32


class SampleParamAddition(link_hooks.TweakLink):

    name = 'NAME'

    def adjust_target(self, cb_args):
        return getattr(cb_args.link, self.target_name)

    def prepare_params(self, link):
        with link.init_scope():
            link.p = chainer.Parameter(
                numpy.random.uniform(-1, 1, SHAPE).astype(DTYPE))

    def deleted(self, link):
        del link.p


class TestExceptions(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(numpy.ones((10, 5), dtype=numpy.float32))
        self.layer = L.Linear(5, 20)

    def test_wrong_weight_name(self):
        wrong_target_name = 'w'
        hook = link_hooks.TweakLink(target_name=wrong_target_name)
        with pytest.raises(ValueError):
            self.layer.add_hook(hook)

    def test_raises_if_used_as_context_manager(self):
        with pytest.raises(NotImplementedError):
            with link_hooks.TweakLink():
                self.layer(self.x)

    def test_raises_if_call_TweakLink(self):
        layer = L.Linear(10, 0).add_hook(link_hooks.TweakLink())
        with pytest.raises(NotImplementedError):
            layer(self.x)

    def check_is_link_initialized(self, lazy=True):
        layer = L.Linear(None if lazy else 10, 10)
        hook = link_hooks.TweakLink()
        layer.add_hook(hook)

        assert hook.is_link_initialized is not lazy

    def test_lazy_initialization(self):
        self.check_is_link_initialized(False)

    def test_initialization(self):
        self.check_is_link_initialized(True)


class TestSampleParamAddition(unittest.TestCase):

    def setUp(self):
        self.x = numpy.ones((10, 5), dtype=numpy.float32)
        self.layer = L.Linear(5, 20)

    def test_added_deleted(self):
        x = self.x.copy()
        layer = self.layer.copy('copy')
        layer.add_hook(SampleParamAddition())

        with chainer.using_config('train', False):
            y1 = layer(x).array

        assert layer.p.shape == SHAPE
        assert layer.p.dtype == DTYPE

        layer.delete_hook(NAME)
        assert getattr(layer, 'p', None) is None

        with chainer.using_config('train', False):
            y2 = layer(x).array

        testing.assert_allclose(y1, y2)


testing.run_module(__name__, __file__)
