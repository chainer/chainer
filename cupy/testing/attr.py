from nose.plugins import attrib

gpu = attrib.attr('gpu')
slow = attrib.attr('slow')


def multi_gpu(gpu_num):
    return attrib.attr(gpu=gpu_num)
