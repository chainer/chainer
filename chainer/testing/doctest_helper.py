import os
import pkg_resources


_gpu_limit = int(os.getenv('CHAINER_TEST_GPU_LIMIT', '-1'))


def skipif(condition):
    # In the readthedocs build, doctest should never be skipped, because
    # otherwise the code would disappear from the documentation.
    if os.environ.get('READTHEDOCS') == 'True':
        return False
    return condition


def skipif_requires_satisfied(*requirements):
    ws = pkg_resources.WorkingSet()
    try:
        ws.require(*requirements)
    except pkg_resources.ResolutionError:
        return False
    return skipif(True)


def skipif_not_enough_cuda_devices(device_count):
    return skipif(0 <= _gpu_limit < device_count)
