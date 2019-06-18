import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing
from chainer import utils
from chainer.utils import type_check

STABILITY_EPS=0.00001

def _euculid_distance(x, x0):
    sql_norm = numpy.sum(numpy.square(x-x0), axis=1)
    return sql_norm

def _fits(x, x0, temp):
    distance_matrix = _euculid_distance(x,x0)
    return numpy.exp(-(distance_matrix/temp))

def _pick(x, x0, temp):
    batch = x.shape[0]
    f = numpy.sum(_fits(x,x0,temp)) - 1
    return f

def _soft_nearest_neighbor_loss(x, y, temp):
    snnl_sum = 0
    batch = len(x)
    for index in range(batch):
        x0 = x[index]
        y0 = y[index]
        denom = _pick(x, x0, temp) 
        x_same = x[y==y0]
        numer = _pick(x_same, x0, temp)
        snnl_sum += -numpy.log(numer / (denom + STABILITY_EPS))

    return snnl_sum / batch

@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
#        {'use_ideep': 'always'},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
#        {'use_chainerx': True, 'chainerx_device': 'native:0'},
#        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
#        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
@testing.parameterize(
#    {'dtype': numpy.float16},
    {'dtype': numpy.float32},
)

class TestSoftNearestNeiborLoss(testing.FunctionTestCase):
    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 5e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 5e-2, 'rtol': 5e-2})

        y = numpy.asarray([0,0,1,1,2,2,3,3,4,4]).astype(numpy.int32)
        temp = numpy.asarray(100).astype(self.dtype)
        self.y = y
        self.temp = temp


    def generate_inputs(self):
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, (10, 5)).astype(dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        loss = functions.soft_nearest_neighbor_loss(x,self.y,self.temp,0)
        return loss,

    def forward_expected(self, inputs):
        x, = inputs

        loss = _soft_nearest_neighbor_loss(x, self.y, self.temp)
        loss = utils.force_array(loss).astype(x.dtype)
        return loss,


testing.run_module(__name__, __file__)
