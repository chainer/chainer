from __future__ import absolute_import

from chainer.configuration import config


_error = None

try:
    import ideep  # NOQA
except ImportError as e:
    _error = e


_SHOULD_USE_IDEEP = {
    '==always': {'always': True, 'auto': False, 'never': False},
    '>=auto':   {'always': True, 'auto': True,  'never': False},
}


def is_available():
    return _error is None


def check_available():
    """Checks if iDeep is available.

    When iDeep is correctly set up, nothing happens.
    Otherwise it raises ``RuntimeError``.
    """
    if _error is not None:
        raise RuntimeError(
            'iDeep is not available.\n'
            'Reason: {}'.format(type(_error).__name__, str(_error)))


def should_use(level):
    """Determines if we should use iDeep.

    This function checks ``chainer.config.use_ideep`` and availability
    of ``ideep`` package.

    Args:
        level (str): iDeep use level. It must be either ``'==always'`` or
            ``'>=auto'``. ``'==always'`` indicates that the ``use_ideep``
            config must be ``'always'`` to use iDeep.

    Returns:
        bool: ``True`` if the caller should use iDeep.

    """
    if not is_available():
        return False

    if level not in _SHOULD_USE_IDEEP:
        raise ValueError('invalid iDeep use level: %s '
                         '(must be either of "==always" or ">=auto")' %
                         repr(level))

    flags = _SHOULD_USE_IDEEP[level]

    use_ideep = config.use_ideep
    if use_ideep not in flags:
        raise ValueError('invalid use_ideep configuration: %s '
                         '(must be either of "always", "auto", or "never")' %
                         repr(use_ideep))
    return flags[use_ideep]
