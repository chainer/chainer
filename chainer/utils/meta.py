import warnings

import six


def final(*args, **kwargs):
    """Decorator to declare a method final.

    By default, :class:`TypeError` is raised when the decorated method is being
    overridden.

    The class in which the decorated method is defined must inherit from a
    base class returned by :meth:`~chainer.utils.enable_final`.

    Args:
        action(type): Specifies what happens when the decorated method is
            being overridden. It can be either an :class:`Exception` class or a
            :class:`Warning` class. :class:`TypeError` by default.
    """
    def wrap(f, action=TypeError):
        assert callable(f)
        f.__override_action = (action,)
        return f

    # apply directly
    if not kwargs and len(args) == 1:
        f, = args
        return wrap(f)

    # with arguments
    assert len(args) == 0

    def w(f):
        return wrap(f, **kwargs)

    return w


class _EnableFinal(type):
    def __new__(cls, name, bases, d):
        for k in d:
            for base in bases:
                f = getattr(base, k, None)  # base method
                if hasattr(f, '__override_action'):
                    action, = getattr(f, '__override_action')
                    if issubclass(action, Warning):
                        # Raise a warning.
                        warnings.warn(
                            'Overriding method {!r}.'.format(k),
                            action)
                    elif issubclass(action, Exception):
                        # Raise error.
                        raise action('method {!r} is final.'.format(k))
                    else:
                        assert False, 'Invalid action: {}'.format(action)
        return super(_EnableFinal, cls).__new__(cls, name, bases, d)


def enable_final(base=(), meta_base=()):
    """Returns a base class in which ``final`` decorator is made available.

    Inheriting from the returned value of this function enables
    :meth:``~chainer.utils.final`` decorator to be applied to the methods of
    the class.

    Args:
        base (type or tuple of types): Base classes of the returned class.
        meta_base (type or tuples of type): Base metaclasses. If any descendant
            classes can directly or indirectly have any metaclasses, these
            metaclasses should be specified here to avoid the metaclass
            conflict.
    """
    if not isinstance(base, (list, tuple)):
        base = (base,)
    if not isinstance(meta_base, (list, tuple)):
        meta_base = (meta_base,)

    base_metaclass = type('base_metaclass', (_EnableFinal,) + meta_base, {})
    return six.with_metaclass(base_metaclass, *base)
