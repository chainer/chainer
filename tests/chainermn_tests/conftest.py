import multiprocessing
import platform
import pytest


def dummy_func():
    pass


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    if int(platform.python_version_tuple()[0]) >= 3:
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process(target=dummy_func)
        p.start()
        p.join()
    yield
