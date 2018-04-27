import unittest

import numpy

from chainer import testing
from chainer import utils


def _setup_tensor(_min, _max, shape, dtype, threshold=None):
    y = numpy.random.uniform(_min, _max, shape).astype(dtype)
    if threshold is not None:
        y[y < threshold] = 0
    return y


@testing.parameterize(*testing.product({
    'shape': [(2, 3), (3, 4)],
    'nbatch': [0, 1, 4],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestCooMatrix(unittest.TestCase):

    def test_to_dense(self):
        if self.nbatch > 0:
            x_shape = (self.nbatch, self.shape[0], self.shape[1])
        else:
            x_shape = self.shape
        x0 = _setup_tensor(.5, 1, x_shape, self.dtype, .75)
        sp_x = utils.to_coo(x0)
        x1 = sp_x.to_dense()
        numpy.testing.assert_array_equal(x0, x1)


testing.run_module(__name__, __file__)
