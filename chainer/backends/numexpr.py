from __future__ import absolute_import

import chainer
from chainer.configuration import config

_numexpr_version = None
_error = None

try:
    import numexpr # NOQA
    _numexpr_version = 0
except ImportError as e:
    _error = e

def is_numexpr_available():
    return _numexpr_version is not None

def check_numexpr_available():
    """Checks if NumExpr is available.

    When NumExpr is correctly set up, nothing happens.
    Otherwise it raises ``RuntimeError``.
    """
    if _numexpr_version is None:
        raise RuntimeError(
            'NumExpr is not available.\n'
            'Reason: {}'.format(type(_error).__name__, str(_error)))

# ------------------------------------------------------------------------------
# numexpr configuration
# ------------------------------------------------------------------------------
_SHOULD_USE_NUMEXPR = {
    '==always': {'always': True, 'auto': False, 'never': False},
    '>=auto':   {'always': True, 'auto': True,  'never': False},
}

def should_use_numexpr(level):
    """Determines if we should use numexpr.

    This function checks ``chainer.config.use_numexpr`` and availability
    of ``numexpr`` package.

    Args:
        level (str): NumExpr use level. It must be either ``'==always'`` or
            ``'>=auto'``. ``'==always'`` indicates that the ``use_numexpr``
            config must be ``'always'`` to use numexpr.

    Returns:
        bool: ``True`` if the caller should use NumExpr.

    """
    if _numexpr_version is None:
        return False

    if level not in _SHOULD_USE_NUMEXPR:
        raise ValueError('invalid NumExpr use level: %s '
                         '(must be either of "==always" or ">=auto")' %
                         repr(level))

    flags = _SHOULD_USE_NUMEXPR[level]

    use_numexpr = config.use_numexpr
    if use_numexpr not in flags:
        raise ValueError('invalid use_numexpr configuration: %s '
                         '(must be either of "always", "auto", or "never")' %
                         repr(use_numexpr))
    return flags[use_numexpr]
