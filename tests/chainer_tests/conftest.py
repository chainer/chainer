from chainer.backends import cuda


def pytest_runtest_teardown(item, nextitem):
    if cuda.available:
        assert cuda.cupy.cuda.runtime.getDevice() == 0


# testing.run_module(__name__, __file__)
