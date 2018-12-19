import functools

import chainer


class cached_property(object):
    """Cache a result of computation of Chainer functions"""

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
