import functools

import chainer


class cached_property(object):
    """Cache a result of computation of Chainer functions

    Caches are stored for each chainer.config.enable_backprop.

    The following example calls ``F.sigmoid`` only once.

    >>> class C(object):
    ...     def __init__(self, x):
    ...         self.x = x
    ...     @chainer.utils.cache.cached_property
    ...     def y(self):
    ...         return F.sigmoid(self.x)
    ...     def loss(self, t):
    ...         return F.mean_squared_error(self.y, t)
    >>> x = chainer.Variable(np.array([2, 3], np.float32))
    >>> obj = C(x)
    >>> loss1 = obj.loss(np.array([0.1, 0.2], np.float32))
    >>> loss2 = obj.loss(np.array([0.3, 0.4], np.float32))

    However, the following example recomputes `obj.y` because the second call
    requires the computational graph.

    >>> with chainer.no_backprop_mode():
    ...     loss1 = obj.loss(np.array([0.1, 0.2], np.float32))
    >>> loss2 = obj.loss(np.array([0.3, 0.4], np.float32))

    """

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        caches = obj.__dict__.setdefault(self.__name__, {})
        backprop_enabled = chainer.config.enable_backprop
        try:
            return caches[backprop_enabled]
        except KeyError:
            value = self.func(obj)
            caches[backprop_enabled] = value
            return value

    def __set__(self, obj, cls):
        # Define __set__ to make cached_property a data descriptor
        raise AttributeError(
            'attribute \'{}\' of {} is readonly'.format(
                self.__name__, cls))
