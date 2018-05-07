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
        if not sp_x.data.shape == sp_x.row.shape == sp_x.col.shape:
            raise ValueError('shape of data/row/col must be identical.')
        max_nnz = self.shape[0] * self.shape[1]
        if self.nbatch > 0:
            if sp_x.data.ndim != 2:
                raise ValueError('ndim of data/row/col must be two.')
            if sp_x.data.shape[0] != self.nbatch:
                raise ValueError('batch size is wrong.')
            if not 0 <= sp_x.data.shape[1] <= max_nnz:
                raise ValueError('num non-zero element is wrong.')
        else:
            if sp_x.data.ndim != 1:
                raise ValueError('ndim of data/row/col must be one.')
            if not 0 <= sp_x.data.shape[0] <= max_nnz:
                raise ValueError('num non-zero element is wrong.')
        x1 = sp_x.to_dense()
        numpy.testing.assert_array_equal(x0, x1)


testing.run_module(__name__, __file__)
