import warnings
import numpy

import cuda
from function import Function

class FunctionSet(object):
    """Set of objects with ``parameters`` and ``gradients`` properties.

    :class:`FunctionSet` is useful to collect parameters and gradients of
    multiple parameterized :class:`Function` objects. :class:`FunctionSet`
    itself also implements :attr:`~FunctionSet.parameters` and
    :attr:`~FunctionSet.gradients`, so it can be nested in another
    :class:`FunctionSet` object.

    Function registration is done by just adding an attribute to
    :class:`FunctionSet` object.

    """
    def __init__(self, **functions):
        """Initializes :class:`FunctionSet` by given key-value pairs of
        :class:`Function` objects.

        Args:
            **functions: ``dict`` of ``str`` key and :class:`Function` values.
                The key-value pairs are just set to the :class:`FunctionSet`
                object as attributes.

        """
        for name, func in functions.iteritems():
            setattr(self, name, func)

    def __setattr__(self, key, value):
        if key == '__collected':
            raise RuntimeError(
                'The name __collected is reserved for internal use')
        self.__dict__[key] = value

    def collect_parameters(self):
        """Returns a tuple of parameters and gradients.

        Returns:
            Tuple (pair) of two tuples. The first element is a tuple of
            parameter arrays, and the second is a tuple of gradient arrays.

        """
        self.__dict__['__collected'] = True
        return self.parameters, self.gradients

    def to_gpu(self, device=None):
        """Migrates all parameters and gradients onto GPU.

        This method calls ``to_gpu`` method of each registered object.

        Args:
            device (int or :class:`pycuda.driver.Device` or ``None``): Device
                ID of GPU. If ``None`` is given, it uses the current device.

        Returns:
            self

        """
        self._warn_if_collected()
        for func in self.__dict__.itervalues():
            if isinstance(func, Function):
                func.to_gpu(device=device)
        return self

    def to_cpu(self):
        """Migrates all parameters and gradients onto CPU.

        This method calls ``to_cpu`` method of each registered object.

        Returns:
            self

        """
        self._warn_if_collected()
        for func in self.__dict__.itervalues():
            if isinstance(func, Function):
                func.to_cpu()
        return self

    def copy_parameters_from(self, params):
        """Copies parameters from another source without reallocation.

        Args:
            params (Iterable): Iterable of parameter arrays.

        """
        for dst, src in zip(self.parameters, params):
            if isinstance(dst, numpy.ndarray):
                if isinstance(src, numpy.ndarray):
                    dst.copy(src)
                else:
                    src.get(dst)
            elif isinstance(src, numpy.ndarray):
                dst.set(src)
            else:
                cuda.copy(src, out=dst)

    @property
    def parameters(self):
        """Tuple of parameter arrays of all registered functions.

        The order of parameters is consistent with :meth:`gradients` property.

        """
        return sum((func.parameters for _, func in self._get_sorted_funcs()), ())

    @parameters.setter
    def parameters(self, params):
        param_iter = iter(params)
        for _, func in self._get_sorted_funcs():
            func.parameters = param_iter

    @property
    def gradients(self):
        """Tuple of gradient arrays of all registered functions.

        The order of gradients is consistent with :meth:`parameters` property.

        """
        return sum((func.gradients for _, func in self._get_sorted_funcs()), ())

    @gradients.setter
    def gradients(self, grads):
        grad_iter = iter(grads)
        for _, func in self._get_sorted_funcs():
            func.gradients = grad_iter

    def _get_sorted_funcs(self):
        return sorted(((k, v) for k, v in self.__dict__.iteritems()
                       if isinstance(v, Function)))

    def _warn_if_collected(self):
        if hasattr(self, '__collected'):
            warnings.warn('A parameter/gradient array owned by an optimizer is'
                          + ' moved over host/devices. It will cause the gradient'
                          + ' updates hidden from the optimizer, which makes the'
                          + ' learning process meaningless. See the issue #47 for'
                          + ' more detailed discussions:'
                          + ' https://github.com/pfnet/chainer/issues/47')
