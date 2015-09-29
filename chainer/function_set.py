import numpy
import six
import warnings

from chainer import cuda
from chainer import link


class FunctionSet(link.DictLink):

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
        """Initializes the function set by given functions.

        Args:
            **functions: ``dict`` of ``str`` key and :class:`Function` values.
                The key-value pairs are just set to the :class:`FunctionSet`
                object as attributes.

        """
        warnings.warn('FunctionSet is deprecated. Use DictLink instead.',
                      DeprecationWarning)
        super(FunctionSet, self).__init__(**functions)

    def __getattr__(self, key):
        return self.children[key]

    def __setattr__(self, key, value):
        if isinstance(value, link.Link):
            self[key] = value
        else:
            super(FunctionSet, self).__setattr__(key, value)

    def __getstate__(self):
        # avoid getattr/setattr
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def collect_parameters(self):
        """Returns a tuple of parameters and gradients.

        Returns:
            Tuple (pair) of two tuples. The first element is a tuple of
            parameter arrays, and the second is a tuple of gradient arrays.

        """
        msg = ("'collect_parameters' is deprecated. "
               "You can pass FunctionSet itself to 'optimizer.setup'")
        warnings.warn(msg, FutureWarning)
        return self

    def copy_parameters_from(self, params):
        """Copies parameters from another source without reallocation.

        Args:
            params (Iterable): Iterable of parameter arrays.

        """
        msg = 'copy_parameters_from is deprecated. Use copyparams instead.'
        for dst, src in zip(self.parameters, params):
            if isinstance(dst, numpy.ndarray):
                if isinstance(src, numpy.ndarray):
                    numpy.copyto(dst, src)
                else:
                    dst[:] = src.get()
            elif isinstance(src, numpy.ndarray):
                dst.set(src)
            else:
                cuda.copy(src, out=dst)

    @property
    def parameters(self):
        """Tuple of parameter arrays of all registered functions.

        The order of parameters is consistent with :meth:`gradients` property.

        """
        tups = {}
        for path, param in self.visitparams():
            tups[path] = param.data
        paths = sorted(tups.keys())
        return tuple(tups[path] for path in paths)

    @parameters.setter
    def parameters(self, params):
        paths = []
        for path, _ in self.visitparams():
            paths.append(path)
        paths.sort()
        d = dict(six.moves.zip(paths, params))

        # replace params by given ones
        for link in self.visitlinks():
            prefix = link._name + '/_params/'
            p = link.params
            for key in p:
                path = prefix + key
                p[key].data = d[path]

    @property
    def gradients(self):
        """Tuple of gradient arrays of all registered functions.

        The order of gradients is consistent with :meth:`parameters` property.

        """
        tups = {}
        for path, param in self.visitparams():
            tups[path] = param.grad
        paths = sorted(tups.keys())
        return tuple(tups[path] for path in paths)

    @gradients.setter
    def gradients(self, grads):
        paths = []
        for path, _ in self.visitparams():
            paths.append(path)
        paths.sort()
        d = dict(six.moves.zip(paths, grads))

        # replace params by given ones
        for link in self.visitlinks():
            prefix = link._name + '/_params/'
            g = link.grads
            for key in g:
                path = prefix + key
                g[key].grad = d[path]
