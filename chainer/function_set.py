import numpy
import six
import warnings

from chainer import cuda
from chainer import link


class FunctionSet(link.DictLink):

    """Set of objects with ``parameters`` and ``gradients`` properties.

    .. deprecated:: v1.4

       It is remained just for backward comatibility.
       **Use** :class:`DictLink` **instead.**

    FunctionSet extends :class:`DictLink` to support APIs compatible to
    previous versions. In versions up to v1.3, it was used to collect
    parameters and gradients of multiple parameterized :class:`Function`
    objects. Now these arrays are managed by the :class:`Link` class and its
    subclasses, so we can use them instead.

    FunctionSet supports object-like interface. A child link can be set just by
    adding an attribute to a FunctionSet object.

    Args:
        **links (dict): Link objects with string keys.

    """
    def __init__(self, **links):
        warnings.warn('FunctionSet is deprecated. Use DictLink instead.',
                      DeprecationWarning)
        super(FunctionSet, self).__init__(**links)

    def __getattr__(self, key):
        """Gets the *child* link of given name."""
        return self.children[key]

    def __setattr__(self, key, value):
        """Sets the link by given name."""
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
        """Returns self.

        .. deprecated:: v1.4
           Pass the :class:`Link` object directly to the :meth:`Optimizer.setup`
           method instead.

        Returns: self.

        """
        msg = ("'collect_parameters' is deprecated. "
               "You can pass FunctionSet itself to 'optimizer.setup'")
        warnings.warn(msg, FutureWarning)
        return self

    def copy_parameters_from(self, params):
        """Copies parameters from another source without reallocation.

        .. deprecated:: v1.4
           Use the :meth:`Link.copyparams` method instead.

        Args:
            params (Iterable): Iterable of parameter arrays.

        """
        msg = 'copy_parameters_from is deprecated. Use copyparams instead.'
        warnings.warn(msg, DeprecationWarning)
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

        .. deprecated:: v1.4
           Use :meth:`Link.visitparams` method instead.

        This property is used for :meth:`copy_parameters_from` method.
        The order of parameters is consistent with :attr:`gradients` property.

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

        .. deprecated:: v1.4
           Use the :meth:`Link.visitparams` method instead.

        The order of gradients is consistent with :attr:`parameters` property.

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
