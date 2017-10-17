from nose.plugins import attrib
from unittest import skip
from chainer import cuda


if cuda.available:
    gpu = attrib.attr('gpu')
else:
    _gpu = attrib.attr('gpu')

    def _skip_attr(op):
        # join decorator skip and attrib.attr
        return skip('gpu not aviable')(_gpu(op))
    gpu = _skip_attr
    
cudnn = attrib.attr('gpu', 'cudnn')
slow = attrib.attr('slow')


def multi_gpu(gpu_num):
    return attrib.attr(gpu=gpu_num)
