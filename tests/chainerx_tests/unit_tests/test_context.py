import pytest

import chainerx


@pytest.fixture
def cache_restore_context(request):
    device = chainerx.get_default_device()
    context = chainerx.get_default_context()
    global_context = chainerx.get_global_default_context()

    def restore_context():
        chainerx.set_global_default_context(global_context)
        chainerx.set_default_context(context)
        chainerx.set_default_device(device)
    request.addfinalizer(restore_context)


def test_creation():
    chainerx.Context()


def test_get_backend():
    context = chainerx.Context()

    backend = context.get_backend('native')
    assert backend.name == 'native'

    assert context.get_backend('native') is backend

    with pytest.raises(chainerx.BackendError):
        context.get_backend('something_that_does_not_exist')


def test_get_device():
    context = chainerx.Context()

    device = context.get_device('native')
    assert device.name == 'native:0'
    assert device.index == 0

    assert context.get_device('native:0') is device
    assert context.get_device('native', 0) is device

    with pytest.raises(chainerx.BackendError):
        context.get_device('something_that_does_not_exist:0')


@pytest.mark.usefixtures('cache_restore_context')
def test_chainerx_get_backend():
    context = chainerx.Context()
    with chainerx.context_scope(context):
        backend = chainerx.get_backend('native')
        assert backend.context is context
        assert backend.name == 'native'


@pytest.mark.usefixtures('cache_restore_context')
def test_chainerx_get_device():
    context = chainerx.Context()
    with chainerx.context_scope(context):
        device = chainerx.get_device('native:0')
        assert device.context is context
        assert device.name == 'native:0'
        assert device is chainerx.get_device('native', 0)
        assert device is chainerx.get_device(device)
        assert chainerx.get_default_device() is chainerx.get_device()


@pytest.mark.usefixtures('cache_restore_context')
def test_default_context():
    context = chainerx.Context()
    global_context = chainerx.Context()

    chainerx.set_global_default_context(None)
    chainerx.set_default_context(None)
    with pytest.raises(chainerx.ContextError):
        chainerx.get_default_context()

    chainerx.set_global_default_context(None)
    chainerx.set_default_context(context)
    assert chainerx.get_default_context() is context

    chainerx.set_global_default_context(global_context)
    chainerx.set_default_context(None)
    assert chainerx.get_default_context() is global_context

    chainerx.set_global_default_context(global_context)
    chainerx.set_default_context(context)
    assert chainerx.get_default_context() is context


@pytest.mark.usefixtures('cache_restore_context')
def test_global_default_context():
    context = chainerx.Context()

    chainerx.set_global_default_context(None)
    with pytest.raises(chainerx.ContextError):
        chainerx.get_global_default_context()

    chainerx.set_global_default_context(context)
    assert chainerx.get_global_default_context() is context


@pytest.mark.usefixtures('cache_restore_context')
def test_context_scope():
    context1 = chainerx.Context()
    context2 = chainerx.Context()

    chainerx.set_default_context(context1)
    with chainerx.context_scope(context2):
        assert chainerx.get_default_context() is context2

    scope = chainerx.context_scope(context2)
    assert chainerx.get_default_context() is context1
    with scope:
        assert chainerx.get_default_context() is context2
    assert chainerx.get_default_context() is context1
