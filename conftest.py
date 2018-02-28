import pytest


def pytest_addoption(parser):
    _add_cuda_option(parser)


def pytest_configure(config):
    _register_cuda_marker(config)


def pytest_runtest_setup(item):
    _setup_cuda_marker(item)


def _add_cuda_option(parser):
    parser.addoption("--cuda", action="store", metavar="LIMIT_NUM", default=-1,
                     help="only run tests that require NVIDIA GPUs less than or equal to the limit. Default: test all")


def _register_cuda_marker(config):
    config.addinivalue_line("markers", "cuda(num=1): mark tests needing the specified number of NVIDIA GPUs.")


def _setup_cuda_marker(item):
    """Pytest marker to indicate number of NVIDIA GPUs required to run the test.

    Tests can be annotated with this decorator (e.g., ``@pytest.mark.cuda``) to
    declare that one NVIDIA GPU is required to run.

    Tests can also be annotated as ``@pytest.mark.cuda(2)`` to declare number of
    NVIDIA GPUs required to run. When running tests, if ``--cuda LIMIT_NUM``
    pytest custom option is set to value greater than or equals to 0, test cases
    that require GPUs more than the limit will be skipped.
    """

    cuda_marker = item.get_marker('cuda')
    if cuda_marker is not None:
        required_num = cuda_marker.args[0] if cuda_marker.args else 1
        minimum_num = int(item.config.getoption('--cuda'))
        if 0 <= minimum_num and minimum_num < required_num:
            pytest.skip('{} NVIDIA GPUs required'.format(required_num))
