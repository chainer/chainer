import os
import pytest
import unittest

_cuda_limit = int(os.getenv('XCHAINER_TEST_CUDA_LIMIT', '-1'))


def multi_cuda(cuda_num):
    """Decorator to indicate number of NVIDIA GPUs required to run the test.

    Tests can be annotated with this decorator (e.g., ``@multi_cuda(2)``) to
    declare number of NVIDIA GPUs required to run. When running tests, if
    ``XCHAINER_TEST_CUDA_LIMIT`` environment variable is set to value greater
    than or equals to 0, test cases that require GPUs more than the limit will
    be skipped.
    """

    return unittest.skipIf(
        0 <= _cuda_limit and _cuda_limit < cuda_num,
        reason='{} NVIDIA GPUs required'.format(cuda_num))


def cuda(f):
    """Decorator to indicate that NVIDIA GPU is required to run the test.

    Tests can be annotated with this decorator (e.g., ``@cuda``) to
    declare that one NVIDIA GPU is required to run.
    """

    return multi_cuda(1)(pytest.mark.cuda(f))
