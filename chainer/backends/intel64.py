from __future__ import absolute_import

import chainer
from chainer.configuration import config


_ideep_version = None
_error = None

try:
    import ideep4py as ideep  # NOQA
    from ideep4py import mdarray  # NOQA
    _ideep_version = 2 if hasattr(ideep, '__version__') else 1
except ImportError as e:
    _error = e
    _ideep_version = None

    class mdarray(object):
        pass  # for type testing


# ------------------------------------------------------------------------------
# ideep configuration
# ------------------------------------------------------------------------------
_SHOULD_USE_IDEEP = {
    '==always': {'always': True, 'auto': False, 'never': False},
    '>=auto':   {'always': True, 'auto': True,  'never': False},
}


def is_ideep_available():
    """Returns if iDeep is available.

    Returns:
        bool: ``True`` if the supported version of iDeep is installed.
    """
    return _ideep_version is not None and _ideep_version == 2


def check_ideep_available():
    """Checks if iDeep is available.

    When iDeep is correctly set up, nothing happens.
    Otherwise it raises ``RuntimeError``.
    """
    if _ideep_version is None:
        # If the error is missing shared object, append a message to
        # redirect to the ideep website.
        msg = str(_error)
        if 'cannot open shared object file' in msg:
            msg += ('\n\nEnsure iDeep requirements are satisfied: '
                    'https://github.com/intel/ideep')
        raise RuntimeError(
            'iDeep is not available.\n'
            'Reason: {}: {}'.format(type(_error).__name__, msg))
    elif _ideep_version != 2:
        raise RuntimeError(
            'iDeep is not available.\n'
            'Reason: Unsupported iDeep version ({})'.format(_ideep_version))


def should_use_ideep(level):
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
    if not is_ideep_available():
        return False

    # TODO(niboshi):
    # Add lowest_version argument and compare with ideep version.
    # Currently ideep does not provide a way to retrieve its version.

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
    """Checks if input arrays are supported for an iDeep primitive.

    Before calling an iDeep primitive (e.g., ``ideep4py.linear.Forward``), you
    need to make sure that all input arrays are ready for the primitive by
    calling this function.
    Information to be checked includes array types, dimesions and data types.
    The function checks ``inputs`` info and ``supported_ndim``.

    Inputs to be tested can be any of ``Variable``, ``numpy.ndarray`` or
    ``ideep4py.mdarray``. However, all inputs to iDeep primitives must be
    ``ideep4py.mdarray``. Callers of iDeep primitives are responsible of
    converting all inputs to ``ideep4py.mdarray``.

    Args:
        inputs (sequence of arrays or variables):
            Inputs to be checked.
        supported_ndim (tuple of ints):
            Supported ndim values for the iDeep primitive.

    Returns:
        bool: ``True`` if all conditions meet.

    """

    def _is_supported_array_type(a):
        return isinstance(a, ideep.mdarray) or ideep.check_type([a])

    if not is_ideep_available():
        return False

    inputs = [x.data if isinstance(x, chainer.variable.Variable)
              else x for x in inputs]

    return (ideep.check_ndim(inputs, supported_ndim)
            and all([_is_supported_array_type(a) for a in inputs]))
