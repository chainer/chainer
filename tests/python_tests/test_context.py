import pytest

import xchainer


@pytest.fixture
def cache_restore_context(request):
    device = xchainer.get_default_device()
    context = xchainer.get_default_context()
    global_context = xchainer.get_global_default_context()

    def restore_context():
        xchainer.set_global_default_context(global_context)
        xchainer.set_default_context(context)
        xchainer.set_default_device(device)
    request.addfinalizer(restore_context)


def test_creation():
    xchainer.Context()


def test_get_backend():
    context = xchainer.Context()

    backend = context.get_backend('native')
    assert backend.name == 'native'

    assert context.get_backend('native') is backend

    with pytest.raises(xchainer.BackendError):
        context.get_backend('something_that_does_not_exist')


def test_get_device():
    context = xchainer.Context()

    device = context.get_device('native')
    assert device.name == 'native:0'
    assert device.index == 0

    assert context.get_device('native:0') is device
    assert context.get_device('native', 0) is device

    with pytest.raises(xchainer.BackendError):
        context.get_device('something_that_does_not_exist:0')


@pytest.mark.usefixtures('cache_restore_context')
def test_default_context():
    context = xchainer.Context()
    global_context = xchainer.Context()

    xchainer.set_global_default_context(None)
    xchainer.set_default_context(None)
    with pytest.raises(xchainer.ContextError):
        xchainer.get_default_context()

    xchainer.set_global_default_context(None)
    xchainer.set_default_context(context)
    assert xchainer.get_default_context() is context

    xchainer.set_global_default_context(global_context)
    xchainer.set_default_context(None)
    assert xchainer.get_default_context() is global_context

    xchainer.set_global_default_context(global_context)
    xchainer.set_default_context(context)
    assert xchainer.get_default_context() is context


@pytest.mark.usefixtures('cache_restore_context')
def test_global_default_context():
    context = xchainer.Context()

    xchainer.set_global_default_context(None)
    with pytest.raises(xchainer.ContextError):
        xchainer.get_global_default_context()

    xchainer.set_global_default_context(context)
    assert xchainer.get_global_default_context() is context


@pytest.mark.usefixtures('cache_restore_context')
def test_context_scope():
    context1 = xchainer.Context()
    context2 = xchainer.Context()

    xchainer.set_default_context(context1)
    with xchainer.context_scope(context2):
        assert xchainer.get_default_context() is context2

    scope = xchainer.context_scope(context2)
    assert xchainer.get_default_context() is context1
    with scope:
        assert xchainer.get_default_context() is context2
    assert xchainer.get_default_context() is context1
