from __future__ import absolute_import

import chainer
from chainer.configuration import config


_ideep_version = None
_error = None

try:
    import ideep4py as ideep  # NOQA
    from ideep4py import mdarray  # NOQA
    _ideep_version = 0
except ImportError as e:
    _error = e

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
    return _ideep_version is not None


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
    if _ideep_version is None:
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
    """Checks if input arrays are supported for ideep optimization.

    Information to be checked includes array types, dimesions and data types.
    The function checks ``inputs`` info and ``supported_ndim``.

    Args:
        inputs (sequence of arrays or variables``):
            Inputs to be checked.
        supported_ndim (tuple of ints):
            Supported ndim values.
            iDeep supports array dimension in either 2 or 4 only.

    Returns:
        bool: ``True`` if all conditions meet.

    """
    if _ideep_version is None:
        return False

    inputs = [x.data if isinstance(x, chainer.variable.Variable)
              else x for x in inputs]

    return (ideep.check_ndim(inputs, supported_ndim)
            and (all([isinstance(a, ideep.mdarray) for a in inputs])
                 or ideep.check_type(inputs)))
