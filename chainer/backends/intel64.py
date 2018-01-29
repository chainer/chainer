from __future__ import absolute_import

from chainer.configuration import config
from chainer import variable


_ideep_version = None
_error = None

try:
    import ideep4py as ideep  # NOQA
    _ideep_version = 0
except ImportError as e:
    _error = e


# ------------------------------------------------------------------------------
# ideep configuration
# ------------------------------------------------------------------------------
_SHOULD_USE_IDEEP = {
    '==always': {'always': True, 'auto': False, 'never': False},
    '>=auto':   {'always': True, 'auto': True,  'never': False},
}


def is_ideep_available():
    return _ideep_version is not None


def check_ideep_available():
    """Checks if iDeep is available.

    When iDeep is correctly set up, nothing happens.
    Otherwise it raises ``RuntimeError``.
    """
    if _ideep_version is None:
        raise RuntimeError(
            'iDeep is not available.\n'
            'Reason: {}'.format(type(_error).__name__, str(_error)))


def should_use_ideep(level, lowest_version=0):
    """Determines if we should use iDeep.

    This function checks ``chainer.config.use_ideep`` and availability
    of ``ideep4py`` package.

    Args:
        level (str): iDeep use level. It must be either ``'==always'`` or
            ``'>=auto'``. ``'==always'`` indicates that the ``use_ideep``
            config must be ``'always'`` to use iDeep.

    Returns:
        bool: ``True`` if the caller should use iDeep.

    """
    if _ideep_version is None:
        return False

    if _ideep_version < lowest_version:
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


def inputs_all_ready(inputs, supported_ndim=(2, 4)):
    """Check inputs and configuration supported for ideep optimization.

    The function checks ``inputs`` info and ``supported_ndim``.

    Args:
        inputs (numpy.ndarray, cupy.ndarray, ideep.mdarray):
            ``inputs`` to be checked including array type, dimension
            and data type.
        supported_ndim: A tuple of ndim. ideep supports array dimension
            in either 2 or 4 only.

    Returns:
        bool: ``True`` if all conditions meet.

    """
    if _ideep_version is None:
        return False

    inputs = [x.data if isinstance(x, variable.Variable)
               else x for x in inputs]

    return (ideep.check_ndim(inputs, supported_ndim)
            and (isinstance(inputs[0], ideep.mdarray)
                 or ideep.check_type(inputs)))
