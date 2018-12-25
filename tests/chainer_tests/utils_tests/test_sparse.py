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
    'use_ldnz': [False, True],
}))
class TestCooMatrix(unittest.TestCase):

    def test_to_dense(self):
        if self.nbatch > 0:
            x_shape = (self.nbatch, self.shape[0], self.shape[1])
        else:
            x_shape = self.shape
        x0 = _setup_tensor(.5, 1, x_shape, self.dtype, .75)
        if self.use_ldnz:
            ldnz = self.shape[0] * self.shape[1]
            sp_x = utils.to_coo(x0, ldnz=ldnz)
        else:
            sp_x = utils.to_coo(x0)
        assert sp_x.data.shape == sp_x.row.shape == sp_x.col.shape
        if self.nbatch > 0:
            assert sp_x.data.ndim == 2
            assert sp_x.data.shape[0] == self.nbatch
            if self.use_ldnz:
                assert sp_x.data.shape[1] == ldnz
            else:
                max_nnz = 0
                for i in range(self.nbatch):
                    max_nnz = max(max_nnz, numpy.count_nonzero(x0[i]))
                assert sp_x.data.shape[1] == max_nnz
        else:
            assert sp_x.data.ndim == 1
            if self.use_ldnz:
                assert sp_x.data.shape[0] == ldnz
            else:
                max_nnz = numpy.count_nonzero(x0)
                assert sp_x.data.shape[0] == max_nnz
        x1 = sp_x.to_dense()
        numpy.testing.assert_array_equal(x0, x1)


@testing.parameterize(*testing.product({
    'shape': [(2, 3), (3, 4)],
    'nbatch': [0, 1, 4],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestGetOrder(unittest.TestCase):

    def test_get_order(self):
        if self.nbatch > 0:
            x_shape = (self.nbatch, self.shape[0], self.shape[1])
        else:
            x_shape = self.shape
        x0 = _setup_tensor(.5, 1, x_shape, self.dtype, .75)
        x0 = numpy.ascontiguousarray(x0)
        if self.nbatch > 0:
            x0[0, 0, self.shape[1]-1] = 1.
            x0[0, self.shape[0]-1, 0] = 1.
        else:
            x0[0, self.shape[1]-1] = 1.
            x0[self.shape[0]-1, 0] = 1.
        sp_x = utils.to_coo(x0)
        row = sp_x.row
        col = sp_x.col
        assert utils.get_order(row, col) == 'C'
        assert utils.get_order(col, row) == 'F'


class TestGetOrder2(unittest.TestCase):

    def test_other_order(self):
        row = numpy.array((0, 2, 4, 1, 3), dtype=numpy.int32)
        col = numpy.array((1, 3, 0, 2, 4), dtype=numpy.int32)
        assert utils.get_order(row, col) == 'other'
        assert utils.get_order(col, row) == 'other'

    def test_diag_order(self):
        row = numpy.array((0, 0, 1, 1, 2), dtype=numpy.int32)
        col = numpy.array((0, 1, 1, 2, 2), dtype=numpy.int32)
        assert utils.get_order(row, col) == 'C'
        assert utils.get_order(col, row) == 'C'

    def test_invalid_shape(self):
        row = numpy.array((0, 2, 4, 1), dtype=numpy.int32)
        col = numpy.array((1, 3, 0, 2, 4), dtype=numpy.int32)
        with self.assertRaises(ValueError):
            utils.get_order(row, col)

    def test_invalid_index_combinatin(self):
        row = numpy.array((0, 2, 4, 1, -1), dtype=numpy.int32)
        col = numpy.array((1, 3, 0, -1, 4), dtype=numpy.int32)
        with self.assertRaises(ValueError):
            utils.get_order(row, col)


testing.run_module(__name__, __file__)
