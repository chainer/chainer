import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Expm1(function.Function):

    @property
    def label(self):
        return 'expm1'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        return utils.force_array(numpy.expm1(x[0])),

    def forward_gpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        return cuda.cupy.expm1(x[0]),

    def backward(self, x, gy):
        y = self.output_data[0]
        return utils.force_array((y + y.dtype.type(1.0)) * gy[0]),


def expm1(x):
    """Elementwise exponential minus one function."""
    return Expm1()(x)
