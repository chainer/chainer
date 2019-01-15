import warnings
from chainer import configuration


def non_deterministic(f_name):
    """Function to warn non-deterministic functions

    If `config.warn_nondeterministic` is True, this function will give a
    warning that this functions contains a non-deterministic function, such
    as atomicAdd.
    """
    if configuration.config.warn_nondeterministic:
        warnings.warn(
            'Non deterministic function of ' + f_name + ' called while'
            ' config.warn_nondeterministic set.')
