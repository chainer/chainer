import numpy

from chainer import functions
from chainer import testing
from chainer import utils

STABILITY_EPS = 0.00001


def _euculid_distance(x, x0):
    sql_norm = numpy.sum(numpy.square(x-x0), axis=1)
    return sql_norm


def _fits(x, x0, temp):
    distance_matrix = _euculid_distance(x, x0)
    return numpy.exp(-(distance_matrix/temp))


def _pick(x, x0, temp):
    f = numpy.sum(_fits(x, x0, temp)) - 1
    return f


def _soft_nearest_neighbor_loss(x, y, temp):
    snnl_sum = 0
    batch = len(x)
    for index in range(batch):
        x0 = x[index]
        y0 = y[index]
        x_same = x[y == y0]
        numer = _pick(x_same, x0, temp)
        denom = _pick(x, x0, temp)
        if len(x_same) == 1:
            snnl_sum += -numpy.log(STABILITY_EPS)
        else:
            snnl_sum += -numpy.log(numer / (denom + STABILITY_EPS))

    return snnl_sum / batch


@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    # ChainerX tests
    + [
    ])
@testing.parameterize(
    *testing.product({
        'shape': [(10, 5)],
        'temp': [numpy.asarray(100)],
        'dtype': [numpy.float16, numpy.float32],
    })
)
@testing.parameterize(
    *testing.product({'enable_double_backprop': [False, True]}))
@testing.fix_random()
class TestSoftNearestNeiborLoss(testing.FunctionTestCase):
    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 5e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 5e-2, 'rtol': 5e-2})
        y = numpy.random.randint(5, size=self.shape[0]).astype(numpy.int32)
        temp = self.temp.astype(self.dtype)
        self.y = y
        self.temp = temp

    def generate_inputs(self):
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, self.shape).astype(dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        y = device.send(self.y)
        temp = device.send(self.temp)
        loss = functions.soft_nearest_neighbor_loss(x, y, temp, 0)
        return loss,

    def forward_expected(self, inputs):
        x, = inputs
        loss = _soft_nearest_neighbor_loss(x, self.y, self.temp)
        loss = utils.force_array(loss).astype(x.dtype)
        return loss,


testing.run_module(__name__, __file__)
