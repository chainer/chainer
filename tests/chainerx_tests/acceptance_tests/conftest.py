from chainerx_tests import cuda_utils


def pytest_ignore_collect(path, config):
    if 0 == cuda_utils.get_cuda_limit():
        if str(path).endswith('_cuda.rst'):
            return True
    return False
