import functools

import numpy

import chainer
from chainer.backends import cuda
from chainer.testing import attr
import chainerx


class BackendConfig(object):

    _props = [
        ('use_chainerx', False),
        ('chainerx_device', None),
        ('use_cuda', False),
        ('use_cudnn', 'never'),
        ('cudnn_deterministic', False),
        ('autotune', False),
        ('use_ideep', 'never'),
        ('cudnn_fast_batch_normalization', False),
    ]

    def __init__(self, params):
        if not isinstance(params, dict):
            raise TypeError('params must be a dict.')
        self._contexts = []

        # Default values
        for k, v in self._props:
            setattr(self, k, v)
        # Specified values
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError('Parameter {} is not defined'.format(k))
            setattr(self, k, v)

        self._check_params()

    def _check_params(self):
        # Checks consistency of parameters

        if self.use_chainerx:
            assert isinstance(self.chainerx_device, str), (
                '\'chainerx_device\' parameter is expected to be a string '
                'representing a ChainerX device specifier')

    @property
    def xp(self):
        if self.use_chainerx:
            return chainerx
        if self.use_cuda:
            return cuda.cupy
        if self.use_ideep:
            return numpy
        return numpy

    def __enter__(self):
        self._contexts = [
            chainer.using_config(
                'use_cudnn', self.use_cudnn),
            chainer.using_config(
                'cudnn_deterministic', self.cudnn_deterministic),
            chainer.using_config(
                'autotune', self.autotune),
            chainer.using_config(
                'use_ideep', self.use_ideep),
        ]
        for c in self._contexts:
            c.__enter__()
        return self

    def __exit__(self, typ, value, traceback):
        for c in reversed(self._contexts):
            c.__exit__(typ, value, traceback)

    def __repr__(self):
        lst = []
        for k, _ in self._props:
            lst.append('{}={!r}'.format(k, getattr(self, k)))
        return '<BackendConfig {}>'.format(' '.join(lst))

    def get_func_str(self):
        """Returns a string that can be used in method name"""
        lst = []
        for k, _ in self._props:
            val = getattr(self, k)
            if val is True:
                val = 'true'
            elif val is False:
                val = 'false'
            else:
                val = str(val)
            lst.append('{}_{}'.format(k, val))
        return '__'.join(lst)

    def get_pytest_marks(self):
        marks = []
        if self.use_chainerx:
            marks.append(attr.chainerx)
            if self.chainerx_device.startswith('cuda:'):
                marks.append(attr.gpu)
        elif self.use_cuda:
            marks.append(attr.gpu)
            if self.use_cudnn != 'never':
                marks.append(attr.cudnn)
        else:
            if self.use_ideep != 'never':
                marks.append(attr.ideep)

        assert all(callable(_) for _ in marks)
        return marks

    def get_array(self, np_array):
        return chainer.backend._obj_to_array(np_array, self._get_array)

    def _get_array(self, np_array):
        if self.use_chainerx:
            # TODO(niboshi): Use backend.to_device or
            # backend.to_chainerx(a, device)
            arr = chainer.backend.to_chainerx(np_array)
            return arr.to_device(self.chainerx_device)
        if self.use_cuda:
            return chainer.backend.cuda.to_gpu(np_array)
        if self.use_ideep:
            return np_array
        return np_array


def _wrap_backend_test_method(impl, param, method_name):
    backend_config = BackendConfig(param)
    marks = backend_config.get_pytest_marks()
    new_method_name = '{}__{}'.format(
        method_name, backend_config.get_func_str())

    @functools.wraps(impl)
    def func(self, *args, **kwargs):
        impl(self, backend_config, *args, **kwargs)

    func.__name__ = new_method_name

    # Apply test marks
    for mark in marks:
        func = mark(func)

    return func, new_method_name


def inject_backend_tests(method_names, params):
    if not (method_names is None or isinstance(method_names, list)):
        raise TypeError('method_names must be either None or a list.')
    if not isinstance(params, list):
        raise TypeError('params must be a list of dicts.')
    if not all(isinstance(d, dict) for d in params):
        raise TypeError('params must be a list of dicts.')

    def wrap(case):
        if method_names is None:
            meth_names = [_ for _ in dir(case) if _.startswith('test_')]
        else:
            meth_names = method_names

        for method_name in meth_names:
            impl = getattr(case, method_name)
            delattr(case, method_name)
            for i_param, param in enumerate(params):
                new_impl, new_method_name = _wrap_backend_test_method(
                    impl, param, method_name)
                if hasattr(case, new_method_name):
                    raise RuntimeError(
                        'Test fixture already exists: {}'.format(
                            new_method_name))
                setattr(case, new_method_name, new_impl)
        return case
    return wrap
