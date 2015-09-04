import numpy
import six
import warnings

from chainer import cuda
from chainer import model


class FunctionSet(model.ModelDict):

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
        msg = 'FunctionSet is deprecated. Use ModelDict instead.'
        warnings.warn(msg, FutureWarning)
        model.ModelDict.__init__(self, **functions)

    def __getattr__(self, key):
        return self.models[key]

    def __setattr__(self, key, value):
        if isinstance(value, model.Model):
            self[key] = value
        else:
            model.ModelDict.__setattr__(self, key, value)

    def __getstate__(self):
        # avoid setattr
        return self.models

    def __setstate__(self, models):
        self.models = models

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
        for path, param, _ in self.visitparams(silent=True):
            tups[path] = param
        paths = sorted(tups.keys())
        return tuple(tups[path] for path in paths)

    @parameters.setter
    def parameters(self, params):
        paths = []
        for path, _, _ in self.visitparams(silent=True):
            paths.append(path)
        paths.sort()
        d = dict(six.moves.zip(paths, params))

        # replace params by given ones
        for model in self.visitmodels():
            prefix = model._name + '/_params/'
            p = model.params
            for key in p:
                path = prefix + key
                p[key] = d[path]

    @property
    def gradients(self):
        """Tuple of gradient arrays of all registered functions.

        The order of gradients is consistent with :meth:`parameters` property.

        """
        tups = {}
        for path, _, grad in self.visitparams():
            tups[path] = grad
        paths = sorted(tups.keys())
        return tuple(tups[path] for path in paths)

    @gradients.setter
    def gradients(self, grads):
        paths = []
        for path, _, _ in self.visitparams():
            paths.append(path)
        paths.sort()
        d = dict(six.moves.zip(paths, grads))

        # replace params by given ones
        for model in self.visitmodels():
            prefix = model._name + '/_params/'
            g = model.grads
            for key in g:
                path = prefix + key
                g[key] = d[path]
